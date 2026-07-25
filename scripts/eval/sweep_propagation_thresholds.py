"""Sweep Component 2A's acceptance thresholds (confidence, cycle-error) to trace out
a coverage-vs-accuracy curve, answering: "how much coverage do we give up to get
higher precision, and where's the useful operating point?"

Efficiency trick: `propagate_keypoints` already computes each candidate's real
confidence and cycle-error before applying the accept/reject gate. So this
script runs propagation ONCE per target with permissive thresholds (nothing
gets rejected), records the raw (confidence, cycle_error, is_correct) triple
for every keypoint, then re-applies many different threshold combinations to
that cached data in plain Python — avoiding re-running the expensive matching
pipeline once per grid point.

Usage:
    python scripts/eval/sweep_propagation_thresholds.py \
        --cub-root ./datasets/cub/CUB_200_2011 \
        --n-classes 10
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from PIL import Image

from evaluate_component2a import build_seed_annotation  # noqa: E402
from data.cub import images_by_class, load_cub  # noqa: E402
from geco.features import load_dinov2, map_point_to_model_space  # noqa: E402
from geco.propagation import propagate_keypoints  # noqa: E402

CONFIDENCE_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
CYCLE_ERROR_GRID = [20.0, 40.0, 60.0, 1e9]  # 1e9 effectively disables the cycle-consistency gate


@dataclass
class RawCandidate:
    confidence: float
    cycle_error: float
    is_correct: bool  # within PCK@alpha of ground truth


def collect_raw_candidates(model, cub_root: str, n_classes: int, n_seeds: int, n_targets: int, alpha: float, seed: int) -> list[RawCandidate]:
    rng = random.Random(seed)
    images = load_cub(cub_root)
    by_class = images_by_class(images)
    eligible = [cls for cls, imgs in by_class.items() if len(imgs) >= n_seeds + n_targets]
    chosen = rng.sample(eligible, min(n_classes, len(eligible)))

    raw: list[RawCandidate] = []
    for cls in chosen:
        imgs = by_class[cls][:]
        rng.shuffle(imgs)
        seed_imgs = imgs[:n_seeds]
        target_imgs = imgs[n_seeds : n_seeds + n_targets]

        seeds = [build_seed_annotation(img) for img in seed_imgs]
        seeds = [s for s in seeds if s.keypoints]
        if not seeds:
            continue

        for target in target_imgs:
            target_img = Image.open(target.path).convert("RGB")
            # Permissive thresholds -> nothing gets rejected, we get every candidate's real stats.
            results = propagate_keypoints(
                model, seeds, target_img, confidence_threshold=0.0, cycle_error_threshold=1e9
            )
            bbox_scale = max(target.bbox[2], target.bbox[3])
            scale = 518 / min(target_img.width, target_img.height)
            threshold = alpha * bbox_scale * scale

            for r in results:
                if r.name not in target.keypoints:
                    continue
                gt = map_point_to_model_space(*target.keypoints[r.name], target_img.width, target_img.height)
                if gt is None:
                    continue
                err = ((r.pixel[0] - gt[0]) ** 2 + (r.pixel[1] - gt[1]) ** 2) ** 0.5
                raw.append(RawCandidate(confidence=r.confidence, cycle_error=r.cycle_error, is_correct=err <= threshold))

        print(f"  class {cls} done ({len(raw)} candidates so far)")

    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-targets-per-class", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading DINOv2...")
    model = load_dinov2()

    print("Collecting raw candidate quality scores (one matching pass per target)...")
    raw = collect_raw_candidates(
        model, args.cub_root, args.n_classes, args.n_seeds, args.n_targets_per_class, args.alpha, args.seed
    )
    print(f"\nCollected {len(raw)} total keypoint candidates. Sweeping thresholds (no recomputation needed)...\n")

    print("=" * 70)
    print(f"{'conf_thresh':<14}{'cycle_thresh':<14}{'coverage':<12}{'accuracy':<12}{'n_accepted':<10}")
    print("-" * 70)
    rows = []
    for conf_t in CONFIDENCE_GRID:
        for cycle_t in CYCLE_ERROR_GRID:
            accepted = [c for c in raw if c.confidence >= conf_t and c.cycle_error <= cycle_t]
            coverage = len(accepted) / len(raw) if raw else 0.0
            accuracy = sum(c.is_correct for c in accepted) / len(accepted) if accepted else 0.0
            rows.append((conf_t, cycle_t, coverage, accuracy, len(accepted)))
            cycle_label = "off" if cycle_t >= 1e9 else f"{cycle_t:.0f}"
            print(f"{conf_t:<14}{cycle_label:<14}{coverage:<12.4f}{accuracy:<12.4f}{len(accepted):<10}")
    print("=" * 70)

    best_balanced = max(rows, key=lambda r: r[2] * r[3])  # coverage * accuracy, a simple balance metric
    print(
        f"\nBest coverage*accuracy balance: conf_thresh={best_balanced[0]}, "
        f"cycle_thresh={best_balanced[1]}  ->  coverage={best_balanced[2]:.4f}, accuracy={best_balanced[3]:.4f}"
    )
    print(
        "\nFor your report: plot accuracy (y) vs coverage (x) across these rows as a "
        "precision-coverage curve — this is the standard way to show a reject-option "
        "system's full operating range instead of one arbitrary threshold pick."
    )


if __name__ == "__main__":
    main()
