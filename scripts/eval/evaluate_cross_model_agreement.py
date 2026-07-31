"""Cross-model agreement vs. self-reported confidence: which one actually
predicts correctness?

METHODOLOGY NOTE (fixed from an earlier version): each model's self-confidence
gap is now scored against that SAME model's OWN correctness -- not against a
single reference model's correctness. Reusing one model's correctness label
for every other model's confidence gap silently turns "does model X know when
IT is wrong" into "does model X's confidence predict a DIFFERENT model's
error," which is a different, weaker, and honestly kind of meaningless
question. This is the standard mistake to avoid when adding models to an
eval like this -- it's an easy one to introduce without noticing, since the
code still runs and produces plausible-looking numbers.

Cross-model agreement is now scored against the ENSEMBLE CENTROID's
correctness (mean of all models' independent predictions) rather than any
one model's correctness. This is the standard deep-ensemble uncertainty
framing (Lakshminarayanan et al., 2017, "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles"): low spread among ensemble
members' predictions should correlate with the ensemble's own accuracy.
It also removes the arbitrary "pick one model as ground truth for the
others" asymmetry entirely.

Usage:
    python scripts/eval/evaluate_cross_model_agreement.py \
        --spair-root ./datasets/spair/SPair-71k --split test \
        --category cat --n-pairs 15 --models dinov2,dino1,clip

The core (build_model_registry, collect_records, compute_gaps) is reused by
sweep_categories.py to run this across every SPair-71k category without
reloading each backbone once per category.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from PIL import Image

from data.spair import load_spair, pairs_by_category  # noqa: E402
from geco.cross_model import mean_pairwise_distance, transfer_and_map  # noqa: E402
from geco.features import IMG_SIZE, load_dinov2  # noqa: E402
from geco.features_clip import IMG_SIZE_CLIP, load_clip  # noqa: E402
from geco.features_dino1 import IMG_SIZE_DINO1, load_dino1  # noqa: E402
from geco.cross_model import geco_match_clip, geco_match_dino1, geco_match_v2  # noqa: E402

AVAILABLE_MODELS = ("dinov2", "dino1", "clip", "dinov3")


def build_model_registry(selected: list[str]):
    registry = {}
    if "dinov2" in selected:
        print("Loading DINOv2...")
        m = load_dinov2()
        registry["dinov2"] = (lambda s, t, m=m: geco_match_v2(m, s, t), IMG_SIZE)
    if "dino1" in selected:
        print("Loading DINOv1 (ungated)...")
        m = load_dino1()
        registry["dino1"] = (lambda s, t, m=m: geco_match_dino1(m, s, t), IMG_SIZE_DINO1)
    if "clip" in selected:
        print("Loading CLIP ViT-B/16 (ungated)...")
        m, p = load_clip()
        registry["clip"] = (lambda s, t, m=m, p=p: geco_match_clip(m, p, s, t), IMG_SIZE_CLIP)
    if "dinov3" in selected:
        print("Loading DINOv3 (requires approved HF access)...")
        from geco.cross_model import geco_match_v3
        from geco.features_v3 import IMG_SIZE_V3, load_dinov3

        m, p = load_dinov3()
        registry["dinov3"] = (lambda s, t, m=m, p=p: geco_match_v3(m, p, s, t), IMG_SIZE_V3)
    return registry


def centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def collect_records(registry: dict, chosen_pairs: list, pck_thresh: float) -> list[dict]:
    """Runs all registered models on every keypoint of every chosen pair.

    Returns records[i] = {"conf": {name: val}, "correct": {name: bool},
                           "agreement_px": float, "ensemble_correct": bool}
    """
    records = []
    for pair in chosen_pairs:
        src_img = Image.open(pair.src_path).convert("RGB")
        trg_img = Image.open(pair.trg_path).convert("RGB")
        results = {name: fn(src_img, trg_img) for name, (fn, _) in registry.items()}

        bbox_scale = max(pair.trg_bbox[2], pair.trg_bbox[3])
        threshold = pck_thresh * bbox_scale

        for (sx, sy), (tx, ty) in zip(pair.src_kps, pair.trg_kps):
            preds, confs, skip = {}, {}, False
            for name, (_, image_size) in registry.items():
                out = transfer_and_map(
                    results[name], image_size, (sx, sy),
                    src_img.width, src_img.height, trg_img.width, trg_img.height,
                )
                if out is None:
                    skip = True
                    break
                trg_orig, conf, _ = out
                preds[name] = trg_orig
                confs[name] = conf
            if skip:
                continue

            correct_by_model = {
                name: (((p[0] - tx) ** 2 + (p[1] - ty) ** 2) ** 0.5) <= threshold
                for name, p in preds.items()
            }
            cx, cy = centroid(list(preds.values()))
            ensemble_correct = (((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5) <= threshold
            agreement_px = mean_pairwise_distance(list(preds.values()))

            records.append({
                "conf": confs,
                "correct": correct_by_model,
                "agreement_px": agreement_px,
                "ensemble_correct": ensemble_correct,
            })
    return records


def compute_gaps(records: list[dict], selected: list[str]) -> dict[str, float]:
    """Returns {signal_name: gap}, e.g. {'dinov2': 0.30, 'dino1': 0.55, ..., 'cross_model_agreement': 0.66}."""
    if not records:
        return {}

    def bucket_pck(get_value, get_correct, reverse: bool) -> float:
        sr = sorted(records, key=get_value, reverse=reverse)
        n = len(sr)
        third = max(1, n // 3)
        top, bottom = sr[:third], sr[-third:]
        top_pck = sum(get_correct(r) for r in top) / len(top)
        bottom_pck = sum(get_correct(r) for r in bottom) / len(bottom)
        return top_pck - bottom_pck

    gaps = {}
    for name in selected:
        gaps[name] = bucket_pck(lambda r, n=name: r["conf"][n], lambda r, n=name: r["correct"][n], True)
    gaps["cross_model_agreement"] = bucket_pck(
        lambda r: r["agreement_px"], lambda r: r["ensemble_correct"], False
    )
    return gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spair-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", default="cat")
    parser.add_argument("--n-pairs", type=int, default=15)
    parser.add_argument("--pck-thresh", type=float, default=0.1)
    parser.add_argument("--models", default="dinov2,dino1,clip", help=f"Comma-separated subset of {AVAILABLE_MODELS}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in AVAILABLE_MODELS]
    if unknown:
        print(f"Unknown model(s) {unknown}, available: {AVAILABLE_MODELS}")
        return
    if len(selected) < 2:
        print("Need at least 2 models to compute cross-model agreement.")
        return

    rng = random.Random(args.seed)
    pairs = load_spair(args.spair_root, split=args.split)
    cat_pairs = pairs_by_category(pairs).get(args.category, [])
    if not cat_pairs:
        print(f"No pairs found for category '{args.category}'.")
        return
    chosen = rng.sample(cat_pairs, min(args.n_pairs, len(cat_pairs)))

    registry = build_model_registry(selected)
    records = collect_records(registry, chosen, args.pck_thresh)

    if not records:
        print("No keypoints scored -- check category/crop coverage.")
        return

    print(f"\nScored {len(records)} keypoints across {len(chosen)} pairs.\n")
    gaps = compute_gaps(records, selected)

    print("=" * 78)
    print(f"{'Signal':<28}{'Judged against':<20}{'Gap':<10}")
    print("-" * 78)
    for name in selected:
        print(f"{name + ' self-confidence':<28}{'own correctness':<20}{gaps[name]:<+10.4f}")
    print(f"{'Cross-model agreement':<28}{'ensemble correctness':<20}{gaps['cross_model_agreement']:<+10.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
