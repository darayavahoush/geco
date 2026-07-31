"""Zero-shot cross-dataset generalization: does the Component 2A propagation
pipeline (built and tuned entirely on CUB birds) work on SPair categories it
has never seen, with a tiny number of real-labeled seed images and ZERO
retraining or fine-tuning?

This reuses `propagate_keypoints` completely unmodified â€” the claim under
test isn't "does confidence-aware matching help" (that's Component 1 /
evaluate_spair.py), it's "does this whole few-shot annotation pipeline
generalize to object categories it's never been tuned on," using only
DINOv2's category-agnostic features + the existing propagation logic.

Since SPair's PairAnnotation only records keypoints visible in BOTH images of
one specific pair, a single image's full keypoint set is scattered across
every pair it appears in â€” aggregate_image_annotations() reconstructs it by
unioning across all pairs in a category (see scripts/data/spair.py).

Usage:
    python scripts/eval/evaluate_spair_zeroshot.py \
        --spair-root ./datasets/spair/SPair-71k \
        --split test \
        --categories bottle,chair,tvmonitor \
        --n-seeds 3 \
        --n-targets-per-category 15 \
        --alpha-thresh 0.1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from PIL import Image

from data.spair import ImageAnnotation, aggregate_image_annotations, load_spair, pairs_by_category  # noqa: E402
from geco.features import load_dinov2, map_point_to_model_space  # noqa: E402
from geco.propagation import SeedAnnotation, propagate_keypoints  # noqa: E402


def build_seed_annotation(ann: ImageAnnotation) -> SeedAnnotation:
    img = Image.open(ann.path).convert("RGB")
    keypoints = {}
    for kp_id, (x, y) in ann.keypoints.items():
        mapped = map_point_to_model_space(x, y, img.width, img.height)
        if mapped is not None:
            keypoints[kp_id] = (round(mapped[0]), round(mapped[1]))
    return SeedAnnotation(image=img, keypoints=keypoints)


def score_predictions(
    predicted: dict[str, tuple[int, int]],
    target: ImageAnnotation,
    target_img: Image.Image,
    alpha_thresh: float,
) -> tuple[int, int]:
    """Returns (n_correct, n_scored) for one target image, same PCK convention
    as evaluate_component2a.py (CUB) and evaluate_spair.py (Component 1)."""
    bbox_scale = max(target.bbox[2], target.bbox[3])
    scale = 518 / min(target_img.width, target_img.height)
    threshold = alpha_thresh * bbox_scale * scale

    n_correct, n_scored = 0, 0
    for kp_id, (px, py) in predicted.items():
        if kp_id not in target.keypoints:
            continue
        gt = map_point_to_model_space(*target.keypoints[kp_id], target_img.width, target_img.height)
        if gt is None:
            continue
        err = ((px - gt[0]) ** 2 + (py - gt[1]) ** 2) ** 0.5
        n_scored += 1
        if err <= threshold:
            n_correct += 1
    return n_correct, n_scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spair-root", required=True)
    parser.add_argument("--split", default="test", choices=["trn", "val", "test"])
    parser.add_argument(
        "--categories",
        default="bottle,chair,tvmonitor",
        help="Comma-separated SPair categories to test zero-shot (default: 3 rigid/mixed "
        "categories with no overlap with CUB birds or anything tuned on so far)",
    )
    parser.add_argument("--n-seeds", type=int, default=3, help="Real-labeled seed images per category")
    parser.add_argument("--n-targets-per-category", type=int, default=15)
    parser.add_argument("--alpha-thresh", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    print(f"Loading SPair-71k '{args.split}' annotations for categories: {categories}")
    all_pairs = load_spair(args.spair_root, split=args.split)
    by_cat = pairs_by_category(all_pairs)

    missing = [c for c in categories if c not in by_cat]
    if missing:
        available = sorted(by_cat.keys())
        print(f"Unknown categories {missing} â€” available categories are: {available}")
        return

    print("Loading DINOv2...")
    model = load_dinov2()

    per_category_results = {}
    total_correct = total_scored = 0

    for cat in categories:
        cat_pairs = by_cat[cat]
        images = aggregate_image_annotations(cat_pairs)
        # Only images with at least one keypoint are usable as seeds or scoreable targets.
        usable = [img for img in images.values() if img.keypoints]
        if len(usable) < args.n_seeds + args.n_targets_per_category:
            print(
                f"  [{cat}] only {len(usable)} usable images, need "
                f"{args.n_seeds + args.n_targets_per_category} â€” skipping"
            )
            continue

        rng.shuffle(usable)
        seed_anns = usable[: args.n_seeds]
        target_anns = usable[args.n_seeds : args.n_seeds + args.n_targets_per_category]

        seeds = [build_seed_annotation(a) for a in seed_anns]
        seeds = [s for s in seeds if s.keypoints]
        if not seeds:
            print(f"  [{cat}] no seeds had in-crop keypoints â€” skipping")
            continue

        cat_correct = cat_scored = 0
        for target in target_anns:
            target_img = Image.open(target.path).convert("RGB")
            results = propagate_keypoints(model, seeds, target_img)
            predicted = {r.name: r.pixel for r in results if r.accepted}
            c, s = score_predictions(predicted, target, target_img, args.alpha_thresh)
            cat_correct += c
            cat_scored += s

        pck = cat_correct / cat_scored if cat_scored else 0.0
        per_category_results[cat] = (pck, cat_scored)
        total_correct += cat_correct
        total_scored += cat_scored
        print(f"  [{cat}] {args.n_seeds} seeds -> {len(target_anns)} targets: PCK@{args.alpha_thresh}={pck:.4f} (n={cat_scored})")

    print("\n" + "=" * 60)
    print(f"{'Category':<20}{'PCK@' + str(args.alpha_thresh):<14}{'n':<8}")
    print("-" * 60)
    for cat, (pck, n) in per_category_results.items():
        print(f"{cat:<20}{pck:<14.4f}{n:<8}")
    overall_pck = total_correct / total_scored if total_scored else 0.0
    print("-" * 60)
    print(f"{'OVERALL (zero-shot)':<20}{overall_pck:<14.4f}{total_scored:<8}")
    print("=" * 60)
    print(
        "Reminder: these categories were NEVER used to build, tune, or select any "
        "hyperparameter in this pipeline (alpha, thresholds, layers, etc. were all "
        "chosen on CUB birds). This measures pure zero-shot transfer of the propagation "
        f"pipeline with only {args.n_seeds} real-labeled seed image(s) per category."
    )


if __name__ == "__main__":
    main()
