"""Cross-model agreement vs. self-reported confidence: which one actually
predicts correctness?

Generalized over any combination of independently-trained backbones:
  - dinov2 : the pipeline's default (DINOv2 ViT-B/14)
  - dino1  : an earlier DINO generation, far less training data, ungated
  - clip   : contrastive image-text supervision, ungated -- the most
             different training objective of the three
  - dinov3 : Meta's newest backbone (gated on HF, requires approval --
             include once your access request clears)

For each SPair-71k keypoint, every selected model independently predicts
where it lands in the target image. This script then compares, as
predictors of correctness against real ground truth:
  1. Each model's own self-reported confidence
  2. Cross-model agreement (mean pairwise pixel distance among all
     selected models' independent predictions -- low distance = high
     agreement)

The research question: does agreement (nobody's self-assessment) beat
every individual model's own introspection?

Usage:
    python scripts/eval/evaluate_cross_model_agreement.py \
        --spair-root ./datasets/spair/SPair-71k --split test \
        --category cat --n-pairs 15 --models dinov2,dino1,clip
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
from geco.cross_model import (  # noqa: E402
    geco_match_clip,
    geco_match_dino1,
    geco_match_v2,
    mean_pairwise_distance,
    transfer_and_map,
)
from geco.features import load_dinov2  # noqa: E402
from geco.features_dino1 import IMG_SIZE_DINO1, load_dino1  # noqa: E402
from geco.features import IMG_SIZE  # noqa: E402
from geco.features_clip import IMG_SIZE_CLIP, load_clip  # noqa: E402

AVAILABLE_MODELS = ("dinov2", "dino1", "clip", "dinov3")


def build_model_registry(selected: list[str]):
    """Returns {name: (match_fn(src_img, trg_img) -> MatchResult, image_size)}."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spair-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", default="cat")
    parser.add_argument("--n-pairs", type=int, default=15)
    parser.add_argument("--pck-thresh", type=float, default=0.1)
    parser.add_argument(
        "--models",
        default="dinov2,dino1,clip",
        help=f"Comma-separated subset of {AVAILABLE_MODELS}",
    )
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

    # records[i] = {"conf": {model_name: value}, "agreement_px": float, "correct": bool}
    records = []

    for i, pair in enumerate(chosen):
        src_img = Image.open(pair.src_path).convert("RGB")
        trg_img = Image.open(pair.trg_path).convert("RGB")

        results = {name: fn(src_img, trg_img) for name, (fn, _) in registry.items()}

        bbox_scale = max(pair.trg_bbox[2], pair.trg_bbox[3])
        threshold = args.pck_thresh * bbox_scale

        for (sx, sy), (tx, ty) in zip(pair.src_kps, pair.trg_kps):
            preds = {}
            confs = {}
            skip = False
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

            agreement_px = mean_pairwise_distance(list(preds.values()))

            # "Correct" judged against dinov2's prediction if present, else the first model's.
            ref_name = "dinov2" if "dinov2" in preds else selected[0]
            ref_pred = preds[ref_name]
            err = ((ref_pred[0] - tx) ** 2 + (ref_pred[1] - ty) ** 2) ** 0.5
            correct = err <= threshold

            records.append({"conf": confs, "agreement_px": agreement_px, "correct": correct})

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(chosen)} pairs done")

    if not records:
        print("No keypoints scored -- check category/crop coverage.")
        return

    print(f"\nScored {len(records)} keypoints across {len(chosen)} pairs "
          f"(correctness judged against '{'dinov2' if 'dinov2' in selected else selected[0]}').\n")

    def bucket_pck(get_value, reverse: bool):
        sorted_recs = sorted(records, key=get_value, reverse=reverse)
        n = len(sorted_recs)
        third = max(1, n // 3)
        top = sorted_recs[:third]
        bottom = sorted_recs[-third:]
        top_pck = sum(r["correct"] for r in top) / len(top)
        bottom_pck = sum(r["correct"] for r in bottom) / len(bottom)
        return top_pck, bottom_pck, top_pck - bottom_pck

    print("=" * 72)
    print(f"{'Signal':<28}{'Top-1/3 PCK':<16}{'Bottom-1/3 PCK':<18}{'Gap':<10}")
    print("-" * 72)
    for name in selected:
        top_pck, bottom_pck, gap = bucket_pck(lambda r, n=name: r["conf"][n], True)
        print(f"{name + ' self-confidence':<28}{top_pck:<16.4f}{bottom_pck:<18.4f}{gap:<+10.4f}")
    top_pck, bottom_pck, gap = bucket_pck(lambda r: r["agreement_px"], False)
    print(f"{'Cross-model agreement':<28}{top_pck:<16.4f}{bottom_pck:<18.4f}{gap:<+10.4f}")
    print("=" * 72)
    print(
        "A bigger gap = that signal separates correct from incorrect matches better.\n"
        "If 'Cross-model agreement' has the biggest gap, agreement between\n"
        "independently-trained models predicts correctness better than any single\n"
        "model's own introspection -- the actual research claim being tested here."
    )


if __name__ == "__main__":
    main()
