"""Evaluate Component 2A (annotation propagation) on CUB-200-2011: naive
nearest-neighbor propagation vs. the full pipeline (confidence-weighted
voting + cycle consistency + RANSAC geometric verification).

For each class: pick a few "seed" images (their real annotations are used as
labels), propagate onto the rest of that class's images, and score against
those images' OWN real annotations (used only for evaluation, never fed to
the propagation pipeline itself).

Usage:
    python scripts/eval/evaluate_component2a.py \
        --cub-root ./datasets/cub/CUB_200_2011 \
        --n-classes 20 \
        --n-seeds 3 \
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
from geco.propagation import SeedAnnotation, propagate_keypoints  # noqa: E402


def build_seed_annotation(cub_img: CubImage) -> SeedAnnotation:
    img = Image.open(cub_img.path).convert("RGB")
    keypoints = {}
    for name, (x, y) in cub_img.keypoints.items():
        mapped = map_point_to_model_space(x, y, img.width, img.height)
        if mapped is not None:
            keypoints[name] = (round(mapped[0]), round(mapped[1]))
    return SeedAnnotation(image=img, keypoints=keypoints)


def score_predictions(predicted: dict[str, tuple[int, int]], target: CubImage, target_img: Image.Image, alpha_thresh: float) -> tuple[int, int]:
    """Returns (n_correct, n_scored) for one target image."""
    bbox_scale = max(target.bbox[2], target.bbox[3])
    scale = 518 / min(target_img.width, target_img.height)
    threshold = alpha_thresh * bbox_scale * scale

    n_correct, n_scored = 0, 0
    for name, (px, py) in predicted.items():
        if name not in target.keypoints:
            continue
        gt = map_point_to_model_space(*target.keypoints[name], target_img.width, target_img.height)
        if gt is None:
            continue
        err = ((px - gt[0]) ** 2 + (py - gt[1]) ** 2) ** 0.5
        n_scored += 1
        if err <= threshold:
            n_correct += 1
    return n_correct, n_scored


def naive_propagate(model, seeds: list[SeedAnnotation], target_img: Image.Image) -> dict[str, tuple[int, int]]:
    """Baseline: single best-confidence seed match per keypoint, no cycle/geometric checks."""
    best: dict[str, tuple[float, tuple[int, int]]] = {}
    for seed in seeds:
        result = geco_match(model, seed.image, target_img)
        for name, (sx, sy) in seed.keypoints.items():
            match = transfer_keypoint(result, sx, sy)
            if name not in best or match.confidence > best[name][0]:
                best[name] = (match.confidence, match.trg_pixel)
    return {name: pixel for name, (conf, pixel) in best.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--n-classes", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=3, help="Seed images per class")
    parser.add_argument("--n-targets-per-class", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Loading CUB-200-2011 annotations...")
    images = load_cub(args.cub_root)
    by_class = images_by_class(images)
    eligible = [
        cls for cls, imgs in by_class.items() if len(imgs) >= args.n_seeds + args.n_targets_per_class
    ]
    chosen_classes = rng.sample(eligible, min(args.n_classes, len(eligible)))

    print("Loading DINOv2...")
    model = load_dinov2()

    naive_correct = naive_scored = 0
    full_correct = full_scored = 0

    for cls in chosen_classes:
        imgs = by_class[cls][:]
        rng.shuffle(imgs)
        seed_imgs = imgs[: args.n_seeds]
        target_imgs = imgs[args.n_seeds : args.n_seeds + args.n_targets_per_class]

        seeds = [build_seed_annotation(img) for img in seed_imgs]
        seeds = [s for s in seeds if s.keypoints]  # drop seeds with no visible/in-crop keypoints
        if not seeds:
            continue

        for target in target_imgs:
            target_img = Image.open(target.path).convert("RGB")

            naive_pred = naive_propagate(model, seeds, target_img)
            c, s = score_predictions(naive_pred, target, target_img, args.alpha)
            naive_correct += c
            naive_scored += s

            full_results = propagate_keypoints(model, seeds, target_img)
            full_pred = {r.name: r.pixel for r in full_results if r.accepted}
            c, s = score_predictions(full_pred, target, target_img, args.alpha)
            full_correct += c
            full_scored += s

        print(f"  class {cls} ({by_class[cls][0].class_name}) done")

    naive_pck = naive_correct / naive_scored if naive_scored else 0.0
    full_pck = full_correct / full_scored if full_scored else 0.0

    print("\n" + "=" * 60)
    print(f"{'Config':<30}{'PCK@' + str(args.alpha):<12}{'n':<8}")
    print("-" * 60)
    print(f"{'Naive (best-confidence only)':<30}{naive_pck:<12.4f}{naive_scored:<8}")
    print(f"{'Full (conf+cycle+geometric)':<30}{full_pck:<12.4f}{full_scored:<8}")
    print("=" * 60)
    print(f"Delta: {full_pck - naive_pck:+.4f}")
    print(
        "Note: the full pipeline also REJECTS low-confidence points (n may differ "
        "between rows) — that's precision/recall being traded, not just raw accuracy."
    )


if __name__ == "__main__":
    main()
