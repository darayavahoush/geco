"""Evaluate Component 1 (the matching engine) on CUB-200-2011 with PCK@alpha,
comparing the confidence-aware pipeline against a vanilla-OT ablation.

PCK@alpha (Percentage of Correct Keypoints): a transferred keypoint counts as
"correct" if it lands within alpha * max(bbox_w, bbox_h) pixels of the true
target location. alpha=0.1 is the standard tight threshold in correspondence
literature.

Usage:
    python scripts/eval/evaluate_component1.py \
        --cub-root ./datasets/cub/CUB_200_2011 \
        --n-pairs 200 \
        --alpha 0.1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from PIL import Image

from data.cub import CubImage, images_by_class, load_cub  # noqa: E402
from geco.features import load_dinov2, map_point_to_model_space  # noqa: E402
from geco.keypoints import transfer_keypoint  # noqa: E402
from geco.matching import geco_match  # noqa: E402

CONFIDENCE_AWARE_KWARGS = dict(alpha=1.0, z_base=0.15, z_range=0.25, reg=0.05)
VANILLA_KWARGS = dict(alpha=0.0, z_base=0.3, z_range=0.0, reg=0.05)  # uniform marginals, flat dustbin


def pose_difficulty(src: CubImage, trg: CubImage) -> float | None:
    """Score how much a shared keypoint LAYOUT differs between src and trg, bbox-normalized.

    This is a proxy for pose/viewpoint/articulation change: if the pairwise
    distances between keypoints (relative to each other) are similar in both
    images, the pose is similar (easy case). If they differ a lot, the bird
    has turned, spread its wings, changed pose, etc. (hard case) — exactly
    where confidence-aware matching is hypothesized to matter most.

    Returns None if fewer than 3 keypoints are shared (not enough to score).
    """
    shared = sorted(set(src.keypoints) & set(trg.keypoints))
    if len(shared) < 3:
        return None

    def normalized_coords(img: CubImage) -> list[tuple[float, float]]:
        bx, by, bw, bh = img.bbox
        scale = max(bw, bh) or 1.0
        return [((img.keypoints[k][0] - bx) / scale, (img.keypoints[k][1] - by) / scale) for k in shared]

    src_pts = normalized_coords(src)
    trg_pts = normalized_coords(trg)

    n = len(shared)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            d_src = ((src_pts[i][0] - src_pts[j][0]) ** 2 + (src_pts[i][1] - src_pts[j][1]) ** 2) ** 0.5
            d_trg = ((trg_pts[i][0] - trg_pts[j][0]) ** 2 + (trg_pts[i][1] - trg_pts[j][1]) ** 2) ** 0.5
            diffs.append(abs(d_src - d_trg))

    return sum(diffs) / len(diffs)


def sample_pairs(images_by_cls: dict[int, list[CubImage]], n_pairs: int, seed: int = 0) -> list[tuple[CubImage, CubImage]]:
    rng = random.Random(seed)
    eligible_classes = [cls for cls, imgs in images_by_cls.items() if len(imgs) >= 2]
    pairs = []
    for _ in range(n_pairs):
        cls = rng.choice(eligible_classes)
        src, trg = rng.sample(images_by_cls[cls], 2)
        pairs.append((src, trg))
    return pairs


def evaluate_config(model, pairs: list[tuple[CubImage, CubImage]], match_kwargs: dict, alpha_thresh: float) -> dict:
    n_correct = 0
    n_total = 0
    for src, trg in pairs:
        src_img = Image.open(src.path).convert("RGB")
        trg_img = Image.open(trg.path).convert("RGB")

        shared_parts = set(src.keypoints) & set(trg.keypoints)
        if not shared_parts:
            continue

        result = geco_match(model, src_img, trg_img, **match_kwargs)
        bbox_scale = max(trg.bbox[2], trg.bbox[3])

        for part in shared_parts:
            sx, sy = src.keypoints[part]
            mapped_src = map_point_to_model_space(sx, sy, src_img.width, src_img.height)
            if mapped_src is None:
                continue

            match = transfer_keypoint(result, round(mapped_src[0]), round(mapped_src[1]))
            if match.is_dustbin:
                n_total += 1  # counts against the model: declined to answer, but ground truth exists
                continue

            # Map the model's 518-space prediction back to original target-image pixels for scoring.
            tx, ty = trg.keypoints[part]
            mapped_trg_gt = map_point_to_model_space(tx, ty, trg_img.width, trg_img.height)
            if mapped_trg_gt is None:
                continue

            pixel_error = (
                (match.trg_pixel[0] - mapped_trg_gt[0]) ** 2 + (match.trg_pixel[1] - mapped_trg_gt[1]) ** 2
            ) ** 0.5
            # Threshold is defined in original-image bbox units; model space is scaled by 518/min(w,h),
            # so rescale the threshold into model space using the target image's own scale factor.
            scale = 518 / min(trg_img.width, trg_img.height)
            threshold_model_space = alpha_thresh * bbox_scale * scale

            n_total += 1
            if pixel_error <= threshold_model_space:
                n_correct += 1

    return {"pck": n_correct / n_total if n_total else 0.0, "n_correct": n_correct, "n_total": n_total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", required=True, help="Path to CUB_200_2011 directory")
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1, help="PCK distance threshold, as fraction of bbox size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stratify-difficulty",
        action="store_true",
        help="Split pairs into easy/medium/hard thirds by pose-layout difference and report each separately.",
    )
    args = parser.parse_args()

    print("Loading CUB-200-2011 annotations...")
    images = load_cub(args.cub_root)
    by_class = images_by_class(images)
    pairs = sample_pairs(by_class, args.n_pairs, seed=args.seed)
    print(f"Sampled {len(pairs)} same-class image pairs.")

    print("Loading DINOv2 (this downloads weights on first run)...")
    model = load_dinov2()

    if not args.stratify_difficulty:
        print("\nEvaluating VANILLA (uniform OT marginals, no confidence weighting)...")
        vanilla = evaluate_config(model, pairs, VANILLA_KWARGS, args.alpha)
        print(f"  PCK@{args.alpha}: {vanilla['pck']:.4f}  ({vanilla['n_correct']}/{vanilla['n_total']})")

        print("\nEvaluating CONFIDENCE-AWARE (this project's pipeline)...")
        conf_aware = evaluate_config(model, pairs, CONFIDENCE_AWARE_KWARGS, args.alpha)
        print(f"  PCK@{args.alpha}: {conf_aware['pck']:.4f}  ({conf_aware['n_correct']}/{conf_aware['n_total']})")

        print("\n" + "=" * 50)
        print(f"{'Config':<20}{'PCK@' + str(args.alpha):<12}{'n':<8}")
        print("-" * 50)
        print(f"{'Vanilla OT':<20}{vanilla['pck']:<12.4f}{vanilla['n_total']:<8}")
        print(f"{'Confidence-aware':<20}{conf_aware['pck']:<12.4f}{conf_aware['n_total']:<8}")
        print("=" * 50)
        delta = conf_aware["pck"] - vanilla["pck"]
        print(f"Delta: {delta:+.4f} ({'improvement' if delta > 0 else 'regression'})")
        return

    # ── Stratified easy/medium/hard evaluation ──────────────────────────────
    scored_pairs = [(pair, pose_difficulty(*pair)) for pair in pairs]
    scored_pairs = [(pair, d) for pair, d in scored_pairs if d is not None]
    scored_pairs.sort(key=lambda x: x[1])

    n = len(scored_pairs)
    third = n // 3
    buckets = {
        "easy (low pose change)": [p for p, _ in scored_pairs[:third]],
        "medium": [p for p, _ in scored_pairs[third : 2 * third]],
        "hard (high pose change)": [p for p, _ in scored_pairs[2 * third :]],
    }

    print(f"\nScored {n} pairs by pose-layout difficulty; split into thirds ({third} pairs each).")
    print("=" * 70)
    print(f"{'Bucket':<26}{'Vanilla PCK':<15}{'Conf-aware PCK':<16}{'Delta':<10}")
    print("-" * 70)
    for name, bucket_pairs in buckets.items():
        if not bucket_pairs:
            continue
        vanilla = evaluate_config(model, bucket_pairs, VANILLA_KWARGS, args.alpha)
        conf_aware = evaluate_config(model, bucket_pairs, CONFIDENCE_AWARE_KWARGS, args.alpha)
        delta = conf_aware["pck"] - vanilla["pck"]
        print(f"{name:<26}{vanilla['pck']:<15.4f}{conf_aware['pck']:<16.4f}{delta:+.4f}")
    print("=" * 70)
    print(
        "If the delta grows from easy -> hard, that supports the hypothesis that "
        "confidence-aware matching specifically helps on harder pose changes, "
        "even if it doesn't help much on an unstratified random sample."
    )


if __name__ == "__main__":
    main()
