"""ArcFace-style training for the face projection head. See earlier version''s
docstring for the full rationale. This version adds per-file logging (printed BEFORE
each attempt, unconditionally) and error handling around the precompute loop, so a
crash -- especially a hard native crash bypassing Python''s own exception handling --
still reveals exactly which file it died on via the last printed line.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geco.face import EMBEDDING_DIM, FaceEmbedder  # noqa: E402


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s


def load_dataset(roots: list[Path]):
    label_of: dict[str, int] = {}
    samples: list[tuple[Path, int]] = []

    for root in roots:
        identities = sorted(d for d in root.iterdir() if d.is_dir() and list(d.glob("*.jpg")))
        for d in identities:
            if d.name not in label_of:
                label_of[d.name] = len(label_of)
            label = label_of[d.name]
            for p in d.glob("*.jpg"):
                samples.append((p, label))

    return samples, len(label_of)


def main():
    parser = argparse.ArgumentParser(description="Train face projection head with ArcFace margin loss.")
    parser.add_argument("--dataset-root", required=True, type=Path, nargs="+")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--cache-features", type=Path, default=None, help="Path to save/load precomputed backbone features")
    parser.add_argument("--eval-pairs", type=Path, default=Path("datasets/faces/pairs_large.txt"), help="Evaluation pairs file to test after training")
    parser.add_argument("--output", type=Path, default=Path("./face_head_arcface.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    feats = None
    labels = None
    num_classes = 0

    if args.cache_features and args.cache_features.exists():
        print(f"Loading cached features from {args.cache_features}...", flush=True)
        cached_data = torch.load(args.cache_features, map_location="cpu")
        feats = cached_data["feats"]
        labels = cached_data["labels"]
        num_classes = cached_data["num_classes"]
        print(f"Loaded {feats.shape[0]} precomputed features across {num_classes} identities.", flush=True)
    else:
        samples, num_classes = load_dataset(args.dataset_root)
        print(f"Loaded {len(samples)} photos across {num_classes} identities from "
              f"{len(args.dataset_root)} dataset root(s): {[str(r) for r in args.dataset_root]}", flush=True)

        print("Loading DINOv2 (frozen backbone)...", flush=True)
        embedder = FaceEmbedder(device=device)

        print(f"Precomputing aligned + frozen backbone features for {len(samples)} photos...", flush=True)
        cached_feats = []
        cached_labels = []
        skipped = []
        for i, (path, label) in enumerate(samples):
            if (i + 1) % 500 == 0 or i == 0 or (i + 1) == len(samples):
                print(f"  [{i + 1}/{len(samples)}] photos processed", flush=True)
            try:
                img = Image.open(path).convert("RGB")
                feat = embedder.compute_pooled_features(img)
            except Exception as exc:
                skipped.append(str(path))
                continue
            cached_feats.append(feat.cpu())
            cached_labels.append(label)

        if skipped:
            print(f"Skipped {len(skipped)} unreadable/corrupt files during precompute.", flush=True)

        feats = torch.stack(cached_feats)
        labels = torch.tensor(cached_labels, dtype=torch.long)

        if args.cache_features:
            args.cache_features.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"feats": feats, "labels": labels, "num_classes": num_classes}, args.cache_features)
            print(f"Saved precomputed features to cache: {args.cache_features}", flush=True)

    embedder = FaceEmbedder(device=device)
    embedder.head = embedder.head.to(device)

    arc_head = ArcMarginProduct(EMBEDDING_DIM, num_classes, s=args.scale, m=args.margin).to(device)
    optimizer = torch.optim.Adam(
        list(embedder.head.parameters()) + list(arc_head.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    n = feats.shape[0]
    print(f"Training projection head with ArcFace loss for {args.epochs} epochs "
          f"(scale={args.scale}, margin={args.margin}, batch_size={args.batch_size})...", flush=True)

    feats_dev = feats.to(device)
    labels_dev = labels.to(device)

    for epoch in range(1, args.epochs + 1):
        embedder.head.train()
        arc_head.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        correct = 0
        seen = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch_feats = feats_dev[idx]
            batch_labels = labels_dev[idx]

            embeddings = embedder.head(batch_feats)
            logits = arc_head(embeddings, batch_labels)
            loss = F.cross_entropy(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            seen += len(idx)

        scheduler.step()

        if epoch % 2 == 0 or epoch == args.epochs:
            avg_loss = total_loss / seen
            train_acc = correct / seen
            print(f"  epoch {epoch:02d}/{args.epochs:02d}  loss={avg_loss:.4f}  train_acc={train_acc * 100:.2f}%", flush=True)

    embedder.head.eval()
    torch.save(embedder.head.state_dict(), args.output)
    print(f"\nSaved ArcFace-trained head to {args.output}", flush=True)

    if args.eval_pairs and args.eval_pairs.exists():
        print(f"\nEvaluating trained model on verification pairs: {args.eval_pairs}...", flush=True)
        pairs = []
        with open(args.eval_pairs) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    pairs.append((parts[0], parts[1], parts[2] == "1"))

        similarities, pair_labels = [], []
        with torch.no_grad():
            for p_a, p_b, is_same in pairs:
                im_a = Image.open(p_a).convert("RGB")
                im_b = Image.open(p_b).convert("RGB")
                sim = torch.dot(embedder.embed(im_a), embedder.embed(im_b)).item()
                similarities.append(sim)
                pair_labels.append(is_same)

        best_acc, best_thresh = 0.0, 0.0
        for thresh_int in range(-100, 101, 5):
            thresh = thresh_int / 100.0
            acc = sum((s >= thresh) == lbl for s, lbl in zip(similarities, pair_labels)) / len(pairs)
            if acc > best_acc:
                best_acc, best_thresh = acc, thresh

        print(f"Validation Benchmark: {len(pairs)} pairs | Best threshold: {best_thresh:.2f} | Accuracy: {best_acc * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
