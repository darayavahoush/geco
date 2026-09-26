"""
Face verification as correspondence-guided anomaly detection (TrustPrint).

Reuses geco.anomaly.detect_anomalies -- Component 2B, UNMODIFIED -- applied to
a new domain: instead of "does this product look like the golden reference,"
the question becomes "does this face look like the reference face." Same-person
pairs are the "normal" class (correspondence should hold, low anomaly score);
different-person pairs are the "anomalous" class (correspondence should break,
high anomaly score). Structurally identical to evaluate_component2b.py's
CUB same-class/different-class proxy, just pointed at YTF identities.

Why this is the interesting result, not just a rerun: detect_anomalies() was
built and tuned for generic object defect-style anomalies (birds, product
photos). If it ALSO separates same/different-person faces well (high AUROC)
with zero face-specific changes, that's evidence the confidence-guided
correspondence signal is a genuinely general "does this match a reference"
primitive -- not something that happened to work for the training domain.
And because anomaly_map is per-patch, a rejected match SHOWS the region that
broke correspondence, instead of returning a bare float like a standard
face-recognition API.

Usage:
    python scripts/eval/evaluate_face_verification_anomaly.py \
        --ytf-data ./face_data/ytf --n-pairs 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from sklearn.metrics import roc_auc_score

from ytf_loader import sample_cross_video_pairs, sample_cross_person_pairs  # noqa: E402
from geco.anomaly import detect_anomalies  # noqa: E402
from geco.features import load_dinov2  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ytf-data", required=True)
    parser.add_argument("--n-pairs", type=int, default=100, help="per class (normal / anomalous)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-crop", action="store_true", help="use full raw frames instead of face-bbox crop, for comparison")
    args = parser.parse_args()

    crop = not args.no_crop
    print(f"Sampling same-person (normal) pairs from YTF... (crop_to_face={crop})")
    normal_pairs = sample_cross_video_pairs(args.ytf_data, n_pairs=args.n_pairs, seed=args.seed, crop_to_face=crop)
    print(f"  got {len(normal_pairs)}")

    print(f"Sampling different-person (anomalous) pairs from YTF... (crop_to_face={crop})")
    anomalous_pairs = sample_cross_person_pairs(args.ytf_data, n_pairs=args.n_pairs, seed=args.seed, crop_to_face=crop)
    print(f"  got {len(anomalous_pairs)}")

    print("Loading DINOv2...")
    model = load_dinov2()

    scores, labels = [], []  # 0 = same person (normal), 1 = different person (anomalous)

    print("\nScoring same-person pairs...")
    for i, (test_img, ref_img) in enumerate(normal_pairs):
        result = detect_anomalies(model, test_img, ref_img)
        scores.append(result.anomaly_score)
        labels.append(0)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(normal_pairs)}")

    print("\nScoring different-person pairs...")
    for i, (test_img, ref_img) in enumerate(anomalous_pairs):
        result = detect_anomalies(model, test_img, ref_img)
        scores.append(result.anomaly_score)
        labels.append(1)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(anomalous_pairs)}")

    if len(set(labels)) < 2:
        raise SystemExit("Need both classes populated to compute AUROC -- check --n-pairs / data availability.")

    auroc = roc_auc_score(labels, scores)
    normal_scores = [s for s, l in zip(scores, labels) if l == 0]
    anomalous_scores = [s for s, l in zip(scores, labels) if l == 1]

    print("\n" + "=" * 60)
    print(f"Mean anomaly score, SAME person pairs:      {sum(normal_scores)/len(normal_scores):.4f}")
    print(f"Mean anomaly score, DIFFERENT person pairs: {sum(anomalous_scores)/len(anomalous_scores):.4f}")
    print(f"AUROC (verification discrimination power):  {auroc:.4f}")
    print("=" * 60)
    print(
        "\nCompare this AUROC against scripts/eval/evaluate_face_verification.py's\n"
        "accuracy (the standard embedding+cosine baseline in geco.face) -- same\n"
        "underlying question, two different mechanisms: one gives a bare\n"
        "similarity float, this one gives a per-patch map of WHERE the two faces\n"
        "disagree (result.anomaly_map / result.dustbin_mask per pair)."
    )


if __name__ == "__main__":
    main()
