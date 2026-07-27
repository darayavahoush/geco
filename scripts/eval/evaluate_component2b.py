"""Evaluate Component 2B (anomaly detection) using CUB-200-2011 as an improvised
benchmark, since no defect-annotated dataset (e.g. MVTec-AD) is downloaded.

The proxy task: SAME-class image pairs stand in for "normal" (should score
LOW anomaly), DIFFERENT-class pairs stand in for "anomalous" (should score
HIGH anomaly) — testing whether the anomaly score actually discriminates
"this looks like the reference" from "this doesn't," which is the entire
premise Track B is built on, without needing pixel-level defect masks.

Reports AUROC: the probability that a random anomalous pair scores higher
than a random normal pair. 0.5 = no discrimination (random), 1.0 = perfect.

Usage:
    python scripts/eval/evaluate_component2b.py \
        --cub-root ./datasets/cub/CUB_200_2011 \
        --n-normal-pairs 100 \
        --n-anomalous-pairs 100
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from PIL import Image
from sklearn.metrics import roc_auc_score

from data.cub import CubImage, images_by_class, load_cub  # noqa: E402
from geco.anomaly import detect_anomalies  # noqa: E402
from geco.features import load_dinov2  # noqa: E402


def sample_normal_pairs(by_class: dict[int, list[CubImage]], n: int, rng: random.Random) -> list[tuple[CubImage, CubImage]]:
    eligible = [cls for cls, imgs in by_class.items() if len(imgs) >= 2]
    pairs = []
    for _ in range(n):
        cls = rng.choice(eligible)
        a, b = rng.sample(by_class[cls], 2)
        pairs.append((a, b))
    return pairs


def sample_anomalous_pairs(by_class: dict[int, list[CubImage]], n: int, rng: random.Random) -> list[tuple[CubImage, CubImage]]:
    classes = list(by_class.keys())
    pairs = []
    for _ in range(n):
        cls_a, cls_b = rng.sample(classes, 2)
        a = rng.choice(by_class[cls_a])
        b = rng.choice(by_class[cls_b])
        pairs.append((a, b))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--n-normal-pairs", type=int, default=100)
    parser.add_argument("--n-anomalous-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("Loading CUB-200-2011 annotations...")
    images = load_cub(args.cub_root)
    by_class = images_by_class(images)

    normal_pairs = sample_normal_pairs(by_class, args.n_normal_pairs, rng)
    anomalous_pairs = sample_anomalous_pairs(by_class, args.n_anomalous_pairs, rng)
    print(f"Sampled {len(normal_pairs)} same-class ('normal') and {len(anomalous_pairs)} different-class ('anomalous') pairs.")

    print("Loading DINOv2...")
    model = load_dinov2()

    scores = []
    labels = []  # 0 = normal, 1 = anomalous

    print("\nScoring normal (same-class) pairs...")
    for i, (test, ref) in enumerate(normal_pairs):
        test_img = Image.open(test.path).convert("RGB")
        ref_img = Image.open(ref.path).convert("RGB")
        result = detect_anomalies(model, test_img, ref_img)
        scores.append(result.anomaly_score)
        labels.append(0)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(normal_pairs)}")

    print("\nScoring anomalous (different-class) pairs...")
    for i, (test, ref) in enumerate(anomalous_pairs):
        test_img = Image.open(test.path).convert("RGB")
        ref_img = Image.open(ref.path).convert("RGB")
        result = detect_anomalies(model, test_img, ref_img)
        scores.append(result.anomaly_score)
        labels.append(1)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(anomalous_pairs)}")

    auroc = roc_auc_score(labels, scores)

    normal_scores = [s for s, l in zip(scores, labels) if l == 0]
    anomalous_scores = [s for s, l in zip(scores, labels) if l == 1]
    mean_normal = sum(normal_scores) / len(normal_scores)
    mean_anomalous = sum(anomalous_scores) / len(anomalous_scores)

    print("\n" + "=" * 60)
    print(f"Mean anomaly score, normal (same-class) pairs:      {mean_normal:.4f}")
    print(f"Mean anomaly score, anomalous (different-class) pairs: {mean_anomalous:.4f}")
    print(f"AUROC (discrimination power):                        {auroc:.4f}")
    print("=" * 60)
    print(
        "AUROC 0.5 = no better than random guessing. 1.0 = perfect separation.\n"
        "This measures whether the score discriminates 'looks like reference' from\n"
        "'doesn't,' as a proxy for real defect detection (no pixel-level anomaly\n"
        "masks available without a dataset like MVTec-AD)."
    )


if __name__ == "__main__":
    main()
