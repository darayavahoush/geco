"""Evaluate a trained face embedding on the standard VERIFICATION-PAIRS protocol
(same format LFW and most face benchmarks use): a list of (image_a, image_b,
is_same_person) triples. Reports accuracy at a swept range of thresholds and
picks the best one -- this is also how you CALIBRATE the threshold used by
geco.face.verify() before trusting it on new photos.

Pairs file format (plain text, one pair per line):
    path/to/photo_a.jpg  path/to/photo_b.jpg  1
    path/to/photo_c.jpg  path/to/photo_d.jpg  0

(1 = same person, 0 = different person -- build this yourself from your
dataset, or adapt LFW's pairs.txt format if using that benchmark directly.)

Usage:
    python scripts/eval/evaluate_face_verification.py \
        --pairs-file ./datasets/faces/pairs.txt \
        --head-weights ./face_head.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from PIL import Image

from geco.face import FaceEmbedder, FaceProjectionHead  # noqa: E402


def load_pairs(pairs_file: Path) -> list[tuple[str, str, bool]]:
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            path_a, path_b, label = parts
            pairs.append((path_a, path_b, label == "1"))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--head-weights", default=None, help="Path to trained head .pt file (omit to use an UNTRAINED head, as a baseline)")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pairs_file))
    if not pairs:
        raise SystemExit(f"No valid pairs found in {args.pairs_file}")
    print(f"Loaded {len(pairs)} verification pairs.")

    print("Loading DINOv2 backbone...")
    head = FaceProjectionHead()
    if args.head_weights:
        head.load_state_dict(torch.load(args.head_weights, map_location="cpu"))
        print(f"Loaded trained head from {args.head_weights}")
    else:
        print("WARNING: using an UNTRAINED head -- this is a baseline only, expect weak results.")
    embedder = FaceEmbedder(head=head)

    similarities = []
    labels = []
    for i, (path_a, path_b, is_same) in enumerate(pairs):
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        emb_a = embedder.embed(img_a)
        emb_b = embedder.embed(img_b)
        sim = torch.dot(emb_a, emb_b).item()
        similarities.append(sim)
        labels.append(is_same)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs scored")

    print(f"\nScored {len(pairs)} pairs. Sweeping thresholds...\n")

    best_acc, best_thresh = 0.0, 0.0
    print("=" * 40)
    print(f"{'threshold':<12}{'accuracy':<12}")
    print("-" * 40)
    for thresh_int in range(-100, 101, 5):
        thresh = thresh_int / 100.0
        correct = sum((sim >= thresh) == label for sim, label in zip(similarities, labels))
        acc = correct / len(pairs)
        marker = ""
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
            marker = " <-- best so far"
        if thresh_int % 20 == 0:
            print(f"{thresh:<12.2f}{acc:<12.4f}{marker}")
    print("=" * 40)
    print(f"\nBest threshold: {best_thresh:.2f}  ->  accuracy: {best_acc:.4f}")
    print(
        f"\nUse this threshold when calling geco.face.verify(embedder, img_a, img_b, "
        f"threshold={best_thresh:.2f}) on new photos."
    )


if __name__ == "__main__":
    main()
