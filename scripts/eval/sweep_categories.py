"""Sweep cross-model agreement vs. self-confidence across EVERY SPair-71k category,
testing the robustness hypothesis directly: individual model confidence can go
NEGATIVE (actively counterproductive) on some categories -- does cross-model
agreement ever do the same, or does it stay reliably positive everywhere?

Loads each backbone ONCE and reuses it across all categories (unlike calling
evaluate_cross_model_agreement.py once per category from the command line,
which would reload every model each time).

Usage:
    python scripts/eval/sweep_categories.py \
        --spair-root ./datasets/spair/SPair-71k --n-pairs 30 --models dinov2,dino1,clip
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from evaluate_cross_model_agreement import (  # noqa: E402
    AVAILABLE_MODELS,
    build_model_registry,
    collect_records,
    compute_gaps,
)
from data.spair import load_spair, pairs_by_category  # noqa: E402

# The 18 standard SPair-71k / PASCAL-derived categories.
ALL_CATEGORIES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "train", "tvmonitor",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spair-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-pairs", type=int, default=30, help="Pairs per category")
    parser.add_argument("--pck-thresh", type=float, default=0.1)
    parser.add_argument("--models", default="dinov2,dino1,clip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--categories", default=None, help="Comma-separated subset; default = all 18")
    args = parser.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in AVAILABLE_MODELS]
    if unknown:
        print(f"Unknown model(s) {unknown}, available: {AVAILABLE_MODELS}")
        return

    categories = (
        [c.strip() for c in args.categories.split(",")] if args.categories else ALL_CATEGORIES
    )

    print("Loading all models once (reused across every category)...")
    registry = build_model_registry(selected)

    rng = random.Random(args.seed)
    pairs = load_spair(args.spair_root, split=args.split)
    by_cat = pairs_by_category(pairs)

    all_gaps: dict[str, dict[str, float]] = {}  # category -> {signal: gap}

    for cat in categories:
        cat_pairs = by_cat.get(cat, [])
        if not cat_pairs:
            print(f"[{cat}] no pairs found, skipping")
            continue
        chosen = rng.sample(cat_pairs, min(args.n_pairs, len(cat_pairs)))
        records = collect_records(registry, chosen, args.pck_thresh)
        if not records:
            print(f"[{cat}] no keypoints scored (crop coverage issue?), skipping")
            continue
        gaps = compute_gaps(records, selected)
        all_gaps[cat] = gaps
        gap_str = "  ".join(f"{k}={v:+.3f}" for k, v in gaps.items())
        print(f"[{cat}] n={len(records)}  {gap_str}")

    if not all_gaps:
        print("No categories produced results.")
        return

    # ── Summary table ────────────────────────────────────────────────────────
    signal_names = selected + ["cross_model_agreement"]
    print("\n" + "=" * (20 + 12 * len(signal_names)))
    header = f"{'Category':<20}" + "".join(f"{s:<12}" for s in signal_names)
    print(header)
    print("-" * (20 + 12 * len(signal_names)))
    for cat, gaps in all_gaps.items():
        row = f"{cat:<20}" + "".join(f"{gaps.get(s, float('nan')):<+12.3f}" for s in signal_names)
        print(row)
    print("=" * (20 + 12 * len(signal_names)))

    # ── Robustness test: how often does each signal go negative? ──────────────
    print("\nRobustness check -- how many categories does each signal go NEGATIVE on?")
    print("(negative = actively counterproductive: trusting this signal makes you MORE likely to accept a bad match)")
    for s in signal_names:
        n_negative = sum(1 for gaps in all_gaps.values() if gaps.get(s, 0) < 0)
        print(f"  {s:<28} negative on {n_negative}/{len(all_gaps)} categories")

    # ── Win rate: how often is each signal the SINGLE BEST? ────────────────────
    print("\nWin rate -- how many categories is each signal the single best predictor?")
    win_counts = {s: 0 for s in signal_names}
    for gaps in all_gaps.values():
        best = max(gaps, key=gaps.get)
        win_counts[best] += 1
    for s in signal_names:
        print(f"  {s:<28} best on {win_counts[s]}/{len(all_gaps)} categories")


if __name__ == "__main__":
    main()
