"""Open-set face IDENTIFICATION (1:N) on top of geco.face verification embeddings.

geco.face answers "are these two photos the same person?". This module answers
"who is this?" against a gallery of enrolled people, including "nobody I know".

Appearance robustness (glasses / beard / both) comes from two places:
  1. the head itself, when trained with scripts/train/train_face_arcface_robust.py;
  2. the gallery: each person keeps SEVERAL templates (every enrolment photo, plus
     optional synthetic glasses/beard variants), and a probe is scored against the
     BEST template per person -- so a bearded probe can match a bearded enrolment
     photo even when the clean-shaven one is far away.

The gallery logic is plain numpy so it can be unit-tested without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

DEFAULT_THRESHOLD = 0.95  # same calibration as geco.face.verify (ArcFace head); recalibrate per head
DEFAULT_MARGIN = 0.01     # best person must beat the runner-up by this much, else "ambiguous"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class NoFaceFound(RuntimeError):
    """Raised when the YuNet detector finds no face (the alignment fallbacks would just guess)."""


# --------------------------------------------------------------------------- gallery


@dataclass
class Match:
    name: str
    score: float          # best template cosine similarity for this person
    n_templates: int
    best_source: str      # which enrolled template matched best (photo path or "synthetic:glasses" ...)


@dataclass
class Result:
    verdict: str          # "match" | "ambiguous" | "unknown"
    name: str | None
    score: float
    margin: float         # best score minus runner-up score (inf if only one person enrolled)
    ranking: list[Match] = field(default_factory=list)


class Gallery:
    """Enrolled people -> a stack of unit-norm embedding templates each."""

    def __init__(self):
        self._emb: list[np.ndarray] = []      # each [D]
        self._person: list[str] = []
        self._source: list[str] = []

    # -- building ---------------------------------------------------------
    def add(self, name: str, embedding: np.ndarray, source: str = "") -> None:
        e = np.asarray(embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(e))
        if n < 1e-8:
            raise ValueError("zero-norm embedding")
        self._emb.append(e / n)
        self._person.append(name)
        self._source.append(source)

    def people(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self._person:
            out[p] = out.get(p, 0) + 1
        return out

    def remove(self, name: str) -> int:
        keep = [i for i, p in enumerate(self._person) if p != name]
        removed = len(self._person) - len(keep)
        self._emb = [self._emb[i] for i in keep]
        self._person = [self._person[i] for i in keep]
        self._source = [self._source[i] for i in keep]
        return removed

    def __len__(self) -> int:
        return len(self._person)

    # -- querying ---------------------------------------------------------
    def rank(self, probe: np.ndarray, top_k: int = 5) -> list[Match]:
        if not self._emb:
            return []
        q = np.asarray(probe, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = np.stack(self._emb) @ q                      # [T]
        best: dict[str, tuple[float, str]] = {}
        count: dict[str, int] = {}
        for sim, person, src in zip(sims, self._person, self._source):
            count[person] = count.get(person, 0) + 1
            if person not in best or sim > best[person][0]:
                best[person] = (float(sim), src)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
        return [Match(name=n, score=s, n_templates=count[n], best_source=src) for n, (s, src) in ranked]

    def identify(self, probe: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
                 margin: float = DEFAULT_MARGIN, top_k: int = 5) -> Result:
        ranking = self.rank(probe, top_k=top_k)
        if not ranking:
            return Result("unknown", None, float("-inf"), float("inf"), [])
        top = ranking[0]
        gap = top.score - ranking[1].score if len(ranking) > 1 else float("inf")
        if top.score < threshold:
            return Result("unknown", None, top.score, gap, ranking)
        if gap < margin:
            return Result("ambiguous", top.name, top.score, gap, ranking)
        return Result("match", top.name, top.score, gap, ranking)

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dim = self._emb[0].shape[0] if self._emb else 0
        np.savez_compressed(
            path,
            emb=np.stack(self._emb) if self._emb else np.zeros((0, dim), np.float32),
            person=np.array(self._person, dtype="U"),
            source=np.array(self._source, dtype="U"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        g = cls()
        path = Path(path)
        if not path.exists():
            return g
        data = np.load(path, allow_pickle=False)
        for e, p, s in zip(data["emb"], data["person"], data["source"]):
            g._emb.append(e.astype(np.float32))
            g._person.append(str(p))
            g._source.append(str(s))
        return g


# --------------------------------------------------------------------------- embedding (needs torch)

EmbedFn = Callable[[Image.Image], tuple[np.ndarray, dict]]  # image -> (embedding, detection meta)


def load_embedder(head_path: str | Path | None = None):
    """FaceEmbedder with a trained head (defaults to the repo's face_head_arcface.pt)."""
    import torch
    from .face import FaceEmbedder, FaceProjectionHead

    root = Path(__file__).resolve().parent.parent.parent
    head_path = Path(head_path) if head_path else root / "face_head_arcface.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"head weights not found: {head_path}")
    head = FaceProjectionHead()
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    head.eval()
    return FaceEmbedder(head=head)


def make_embed_fn(embedder, require_detection: bool = True) -> EmbedFn:
    """Wrap a FaceEmbedder: detect+align, refuse to guess when no face is found."""
    from .face_align import extract_face_bbox_and_crop

    def embed(img: Image.Image) -> tuple[np.ndarray, dict]:
        crop, meta = extract_face_bbox_and_crop(img)
        if require_detection and meta.get("method") != "yunet_dnn":
            raise NoFaceFound("no face detected (detector fell back to a guessed crop)")
        return embedder.embed_aligned(crop).cpu().numpy(), meta

    return embed


def synthetic_variants(embedder, aligned: Image.Image, modes: Sequence[str] = ("glasses", "beard", "both"),
                       seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """Embeddings of synthetic glasses/beard versions of an aligned crop (for gallery enrolment)."""
    from .appearance_aug import random_appearance_change

    rng = np.random.default_rng(seed)
    out = []
    for m in modes:
        img, tag = random_appearance_change(aligned, rng, mode=m)
        out.append((f"synthetic:{tag}", embedder.embed_aligned(img).cpu().numpy()))
    return out


def list_images(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def enroll_photos(gallery: Gallery, embedder, name: str, photos: Sequence[Path],
                  synthetic_per_photo: int = 0, seed: int = 0) -> tuple[int, list[str]]:
    """Add every photo (and optionally synthetic appearance variants) as templates for `name`.
    Returns (templates_added, skipped_messages)."""
    from .face_align import extract_face_bbox_and_crop

    added, skipped = 0, []
    for i, p in enumerate(photos):
        try:
            crop, meta = extract_face_bbox_and_crop(Image.open(p).convert("RGB"))
            if meta.get("method") != "yunet_dnn":
                raise NoFaceFound("no face detected")
            gallery.add(name, embedder.embed_aligned(crop).cpu().numpy(), source=str(p))
            added += 1
            if synthetic_per_photo > 0:
                modes = ("glasses", "beard", "both")[:max(1, min(3, synthetic_per_photo))]
                for tag, emb in synthetic_variants(embedder, crop, modes, seed=seed + i):
                    gallery.add(name, emb, source=f"{tag}<-{p.name}")
                    added += 1
        except Exception as exc:  # keep going; report at the end
            skipped.append(f"{p}: {exc}")
    return added, skipped
