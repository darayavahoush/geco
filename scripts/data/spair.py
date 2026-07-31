"""Loader for the SPair-71k dataset: pixel-level, index-matched keypoint pairs
with real annotated difficulty flags (viewpoint_variation, scale_variation,
truncation, occlusion) â€” unlike CUB, where difficulty had to be approximated
from bounding-box-derived proxies.

Expects the standard SPair-71k layout:
    <spair_root>/JPEGImages/<category>/<name>.jpg
    <spair_root>/PairAnnotation/<split>/<pair_id>-<src>-<trg>:<category>.json

Each annotation JSON looks like:
    {"pair_id": 1, "src_imname": "2008_002719.jpg", "trg_imname": "2008_004100.jpg",
     "src_imsize": [500, 333, 3], "trg_imsize": [500, 332, 3],
     "src_bndbox": [210, 34, 475, 250], "trg_bndbox": [36, 92, 500, 200],
     "category": "aeroplane", "src_kps": [[212, 153], ...], "trg_kps": [[39, 145], ...],
     "kps_ids": ["0", "7", "11"], "mirror": 0, "viewpoint_variation": 0,
     "scale_variation": 0, "truncation": 2, "occlusion": 2}

Note bndbox is [x1, y1, x2, y2] (corners), NOT [x, y, w, h] like CUB's format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpairPair:
    pair_id: int
    category: str
    src_path: Path
    trg_path: Path
    src_kps: list[tuple[float, float]]
    trg_kps: list[tuple[float, float]]
    kps_ids: list[str]
    src_bbox: tuple[float, float, float, float]  # (x, y, w, h)
    trg_bbox: tuple[float, float, float, float]
    src_size: tuple[int, int]  # (w, h)
    trg_size: tuple[int, int]
    mirror: bool
    viewpoint_variation: int
    scale_variation: int
    truncation: int
    occlusion: int

    @property
    def hardness(self) -> int:
        """Sum of the four real annotated difficulty flags. Higher = harder pair."""
        return (
            self.viewpoint_variation + self.scale_variation + self.truncation + self.occlusion
        )


def _bndbox_to_xywh(bndbox: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bndbox
    return (x1, y1, x2 - x1, y2 - y1)


def load_spair(spair_root: str | Path, split: str = "test") -> list[SpairPair]:
    """Load every pair annotation for one split (trn / val / test)."""
    root = Path(spair_root)
    ann_dir = root / "PairAnnotation" / split
    if not ann_dir.exists():
        raise FileNotFoundError(
            f"No PairAnnotation/{split} directory under {root} \u2014 check --spair-root "
            "and that the archive was fully extracted."
        )

    pairs: list[SpairPair] = []
    for json_path in sorted(ann_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)

        category = data["category"]
        img_dir = root / "JPEGImages" / category

        pairs.append(
            SpairPair(
                pair_id=data["pair_id"],
                category=category,
                src_path=img_dir / data["src_imname"],
                trg_path=img_dir / data["trg_imname"],
                src_kps=[(float(x), float(y)) for x, y in data["src_kps"]],
                trg_kps=[(float(x), float(y)) for x, y in data["trg_kps"]],
                kps_ids=list(data["kps_ids"]),
                src_bbox=_bndbox_to_xywh(data["src_bndbox"]),
                trg_bbox=_bndbox_to_xywh(data["trg_bndbox"]),
                src_size=(data["src_imsize"][0], data["src_imsize"][1]),
                trg_size=(data["trg_imsize"][0], data["trg_imsize"][1]),
                mirror=bool(data.get("mirror", 0)),
                viewpoint_variation=int(data.get("viewpoint_variation", 0)),
                scale_variation=int(data.get("scale_variation", 0)),
                truncation=int(data.get("truncation", 0)),
                occlusion=int(data.get("occlusion", 0)),
            )
        )
    return pairs


def pairs_by_category(pairs: list[SpairPair]) -> dict[str, list[SpairPair]]:
    by_cat: dict[str, list[SpairPair]] = {}
    for p in pairs:
        by_cat.setdefault(p.category, []).append(p)
    return by_cat


def hardness_terciles(pairs: list[SpairPair]) -> tuple[int, int]:
    """Data-driven easy/medium/hard cutoffs (33rd / 66th percentile of `hardness`).

    SPair's flag values aren't a fixed 0/1 scale (e.g. truncation/occlusion can be
    0, 1, or 2), so hardcoded thresholds would be arbitrary â€” terciles adapt to
    whatever distribution this split/category subset actually has.
    """
    values = sorted(p.hardness for p in pairs)
    n = len(values)
    lo = values[int(n * 0.33)]
    hi = values[int(n * 0.66)]
    return lo, hi


def bucket_for(pair: SpairPair, lo: int, hi: int) -> str:
    if pair.hardness <= lo:
        return "easy"
    elif pair.hardness <= hi:
        return "medium"
    return "hard"


@dataclass
class ImageAnnotation:
    path: Path
    keypoints: dict[str, tuple[float, float]]  # kps_id -> (x, y) in original image space
    bbox: tuple[float, float, float, float]  # (x, y, w, h)
    size: tuple[int, int]  # (w, h)


def aggregate_image_annotations(pairs: list[SpairPair]) -> dict[Path, ImageAnnotation]:
    """Reconstruct full per-image keypoint sets from pair annotations.

    Each PairAnnotation JSON only lists keypoints visible in BOTH members of that
    specific pair, so any single image's full keypoint set is scattered across every
    pair it appears in. This scans every pair an image participates in (as either
    src or trg) and unions the keypoints seen for it, since the same physical
    keypoint has the same pixel location in that image regardless of which pair
    it was annotated alongside.

    This is a substitute for the (unparsed) ImageAnnotation/ directory, built purely
    from PairAnnotation data we already know how to read.
    """
    images: dict[Path, ImageAnnotation] = {}

    def _update(path: Path, kps: list[tuple[float, float]], ids: list[str],
                bbox: tuple[float, float, float, float], size: tuple[int, int]) -> None:
        if path not in images:
            images[path] = ImageAnnotation(path=path, keypoints={}, bbox=bbox, size=size)
        images[path].keypoints.update(dict(zip(ids, kps)))

    for p in pairs:
        _update(p.src_path, p.src_kps, p.kps_ids, p.src_bbox, p.src_size)
        _update(p.trg_path, p.trg_kps, p.kps_ids, p.trg_bbox, p.trg_size)

    return images
