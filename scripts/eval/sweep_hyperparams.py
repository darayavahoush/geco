"""Sweep (alpha, z_base) on a fixed pair sample to check whether the confidence-aware
config's default hyperparameters were simply untuned, masking a real effect.

Usage:
    python scripts/eval/sweep_hyperparams.py --cub-root ./datasets/cub/CUB_200_2011 --n-pairs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from evaluate_component1 import evaluate_config, sample_pairs, VANILLA_KWARGS  # noqa: E402
from data.cub import images_by_class, load_cub  # noqa: E402
from geco.features import load_dinov2  # noqa: E402

ALPHA_GRID = [0.3, 0.5, 0.8, 1.0]
Z_BASE_GRID = [0.15, 0.3, 0.45]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--pck-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading CUB-200-2011 annotations...")
    images = load_cub(args.cub_root)
    by_class = images_by_class(images)
    pairs = sample_pairs(by_class, args.n_pairs, seed=args.seed)
    print(f"Sampled {len(pairs)} pairs (fixed across the whole sweep for a fair comparison).")

    print("Loading DINOv2...")
    model = load_dinov2()

    print("\nBaseline (vanilla OT)...")
    vanilla = evaluate_config(model, pairs, VANILLA_KWARGS, args.pck_alpha)
    print(f"  PCK@{args.pck_alpha}: {vanilla['pck']:.4f}")

    print(f"\nSweeping {len(ALPHA_GRID)}x{len(Z_BASE_GRID)} = {len(ALPHA_GRID) * len(Z_BASE_GRID)} configs...")
    print("=" * 60)
    print(f"{'alpha':<10}{'z_base':<10}{'PCK@' + str(args.pck_alpha):<12}{'vs vanilla':<12}")
    print("-" * 60)

    best = (None, -1.0)
    for a in ALPHA_GRID:
        for zb in Z_BASE_GRID:
            kwargs = dict(alpha=a, z_base=zb, z_range=0.25, reg=0.05)
            result = evaluate_config(model, pairs, kwargs, args.pck_alpha)
            delta = result["pck"] - vanilla["pck"]
            print(f"{a:<10}{zb:<10}{result['pck']:<12.4f}{delta:+.4f}")
            if result["pck"] > best[1]:
                best = ((a, zb), result["pck"])

    print("=" * 60)
    print(f"Best config: alpha={best[0][0]}, z_base={best[0][1]}  ->  PCK@{args.pck_alpha}={best[1]:.4f}")
    print(f"Vanilla baseline: {vanilla['pck']:.4f}")
    print(f"Best improvement over vanilla: {best[1] - vanilla['pck']:+.4f}")


if __name__ == "__main__":
    main()
