"""Evaluate Component 1 (confidence-aware OT matching) on SPair-71k: vanilla
OT (alpha=0, uniform marginals) vs. the tuned confidence-aware config
(alpha=1.0 by default, from the earlier 12-config sweep on CUB), stratified
by SPair's REAL annotated difficulty flags (viewpoint_variation,
scale_variation, truncation, occlusion) instead of the improvised
bbox-based proxy used for CUB.

Unlike CUB (Component 2A, which propagates from seed annotations onto
unlabeled images), SPair-71k ships pairs with already index-matched
src/trg keypoints, so this evaluates matching directly: for each pair, run
the model once per alpha config, transfer every source keypoint, and score
against the real target keypoint at the same index. No propagation,
cycle-consistency, or geometric verification involved here â€” this isolates
Component 1 (the matching engine itself), not Component 2A.

Usage:
    python scripts/eval/evaluate_spair.py \
        --spair-root ./datasets/spair/SPair-71k \
        --split test \
        --n-pairs 200 \
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

from data.spair import SpairPair, bucket_for, hardness_terciles, load_spair  # noqa: E402
from geco.features import load_dinov2, map_point_to_model_space  # noqa: E402
from geco.keypoints import transfer_keypoint  # noqa: E402
from geco.matching import geco_match  # noqa: E402


def score_pair(model, pair: SpairPair, alpha: float, alpha_thresh: float) -> tuple[int, int]:
    """Returns (n_correct, n_scored) for one pair at one alpha config.

    PCK threshold uses the TARGET bbox (standard SPair-71k per-image PCK
    protocol), scaled into the model's 518-space the same way evaluate_component2a.py
    scales CUB's bbox.
    """
    src_img = Image.open(pair.src_path).convert("RGB")
    trg_img = Image.open(pair.trg_path).convert("RGB")

    result = geco_match(model, src_img, trg_img, alpha=alpha)

    trg_scale = 518 / min(trg_img.width, trg_img.height)
    bbox_scale = max(pair.trg_bbox[2], pair.trg_bbox[3])
    threshold = alpha_thresh * bbox_scale * trg_scale

    n_correct, n_scored = 0, 0
    for (sx, sy), (tx, ty) in zip(pair.src_kps, pair.trg_kps):
        src_mapped = map_point_to_model_space(sx, sy, src_img.width, src_img.height)
        trg_mapped = map_point_to_model_space(tx, ty, trg_img.width, trg_img.height)
        if src_mapped is None or trg_mapped is None:
            continue  # keypoint fell outside the model's crop

        match = transfer_keypoint(result, *src_mapped)
        err = (
            (match.trg_pixel[0] - trg_mapped[0]) ** 2 + (match.trg_pixel[1] - trg_mapped[1]) ** 2
        ) ** 0.5

        n_scored += 1
        if err <= threshold:
            n_correct += 1

    return n_correct, n_scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spair-root", required=True)
    parser.add_argument("--split", default="test", choices=["trn", "val", "test"])
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--alpha-vanilla", type=float, default=0.0)
    parser.add_argument(
        "--alpha-full", type=float, default=1.0, help="Tuned value from the CUB 12-config sweep"
    )
    parser.add_argument("--alpha-thresh", type=float, default=0.1, help="PCK@alpha threshold")
    parser.add_argument(
        "--category", default=None, help="Restrict to one SPair category (default: all)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading SPair-71k '{args.split}' annotations...")
    pairs = load_spair(args.spair_root, split=args.split)
    if args.category:
        pairs = [p for p in pairs if p.category == args.category]
    if not pairs:
        print("No pairs matched \u2014 check --spair-root / --split / --category.")
        return

    chosen = rng.sample(pairs, min(args.n_pairs, len(pairs)))
    lo, hi = hardness_terciles(pairs)  # cutoffs from the FULL split, not just the sample
    print(
        f"Loaded {len(pairs)} pairs, evaluating on {len(chosen)}. "
        f"Hardness terciles (sum of viewpoint/scale/truncation/occlusion flags): "
        f"easy<={lo}, medium<={hi}, hard>{hi}"
    )

    print("Loading DINOv2...")
    model = load_dinov2()

    buckets = ["easy", "medium", "hard", "overall"]
    configs = {"vanilla": args.alpha_vanilla, "full": args.alpha_full}
    counts = {c: {b: [0, 0] for b in buckets} for c in configs}

    for i, pair in enumerate(chosen):
        bucket = bucket_for(pair, lo, hi)
        for cfg_name, alpha in configs.items():
            c, s = score_pair(model, pair, alpha, args.alpha_thresh)
            counts[cfg_name][bucket][0] += c
            counts[cfg_name][bucket][1] += s
            counts[cfg_name]["overall"][0] += c
            counts[cfg_name]["overall"][1] += s

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(chosen)} pairs done")

    def pck(cfg, bucket):
        c, s = counts[cfg][bucket]
        return (c / s if s else 0.0), s

    print("\n" + "=" * 72)
    header = f"{'Bucket':<10}{'Vanilla PCK@' + str(args.alpha_thresh):<22}"
    header += f"{'Full PCK@' + str(args.alpha_thresh):<22}{'Delta':<10}{'n':<8}"
    print(header)
    print("-" * 72)
    for bucket in buckets:
        v_pck, v_n = pck("vanilla", bucket)
        f_pck, f_n = pck("full", bucket)
        print(f"{bucket:<10}{v_pck:<22.4f}{f_pck:<22.4f}{f_pck - v_pck:<+10.4f}{f_n:<8}")
    print("=" * 72)
    print(
        "Note: 'n' is the full config's scored-keypoint count; vanilla scores the "
        "same keypoints (transfer_keypoint never rejects \u2014 only propagation does), "
        "so counts should match unless a keypoint fell outside the model's crop for "
        "one config."
    )


if __name__ == "__main__":
    main()
