"""Diagnostic: print the raw confidence / cycle-error / geometric-inlier values
behind one propagate_keypoints() call, to see WHY every candidate is getting
rejected for a given category (rather than just observing that n=0).

Usage:
    python scripts/eval/diagnose_propagation.py \
        --spair-root ./datasets/spair/SPair-71k --split test --category bottle
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from PIL import Image

from data.spair import aggregate_image_annotations, load_spair, pairs_by_category  # noqa: E402
from geco.features import load_dinov2, map_point_to_model_space  # noqa: E402
from geco.matching import geco_match  # noqa: E402
from geco.propagation import SeedAnnotation, propagate_keypoints  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spair-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", required=True)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pairs = load_spair(args.spair_root, split=args.split)
    by_cat = pairs_by_category(pairs)
    cat_pairs = by_cat[args.category]
    images = aggregate_image_annotations(cat_pairs)
    usable = [img for img in images.values() if img.keypoints]
    print(f"Category '{args.category}': {len(usable)} usable images, "
          f"keypoint counts per image: {sorted(len(img.keypoints) for img in usable)}")

    rng.shuffle(usable)
    seed_anns = usable[: args.n_seeds]
    target_ann = usable[args.n_seeds]

    print("Loading DINOv2...")
    model = load_dinov2()

    seeds = []
    for ann in seed_anns:
        img = Image.open(ann.path).convert("RGB")
        kps = {}
        for kp_id, (x, y) in ann.keypoints.items():
            mapped = map_point_to_model_space(x, y, img.width, img.height)
            if mapped is not None:
                kps[kp_id] = (round(mapped[0]), round(mapped[1]))
        seeds.append(SeedAnnotation(image=img, keypoints=kps))
        print(f"  seed {ann.path.name}: {len(kps)} in-crop keypoints")

    target_img = Image.open(target_ann.path).convert("RGB")
    print(f"  target {target_ann.path.name}: {len(target_ann.keypoints)} real keypoints")

    # Raw match diagnostics: mean patch confidence + entropy for each seed->target match
    print("\n--- raw geco_match() diagnostics per seed ---")
    for i, seed in enumerate(seeds):
        result = geco_match(model, seed.image, target_img)
        print(f"  seed {i}: mean_confidence={result.mean_confidence:.4f}  entropy={result.entropy:.4f}")

    print("\n--- propagate_keypoints() per-keypoint outcome ---")
    results = propagate_keypoints(model, seeds, target_img)
    results.sort(key=lambda r: -r.confidence)
    for r in results:
        print(
            f"  {r.name:<8} conf={r.confidence:.4f}  cycle_error={r.cycle_error:7.2f}px  "
            f"geo_inlier={r.geometric_inlier!s:<5}  votes={r.n_votes}  accepted={r.accepted}"
        )

    n_accepted = sum(1 for r in results if r.accepted)
    print(f"\n{n_accepted}/{len(results)} keypoints accepted "
          f"(thresholds: confidence>=0.10, cycle_error<=40px)")


if __name__ == "__main__":
    main()
