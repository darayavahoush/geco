"""
Sweep geco.matching.geco_match's `reg` (Sinkhorn entropic regularization) for
the face-verification-as-anomaly task. Default reg=0.05 was failing to
converge on every single pair in evaluate_face_verification_anomaly.py --
this checks whether a value that actually converges also verifies better,
or whether the non-convergence wasn't hurting discrimination anyway.

Samples pairs ONCE, reuses them across every reg value, so the comparison
isn't confounded by different random pairs per setting.

Usage:
    python scripts/eval/sweep_face_reg.py --ytf-data ..\\face_data\\ytf --n-pairs 50
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from sklearn.metrics import roc_auc_score

from ytf_loader import sample_cross_video_pairs, sample_cross_person_pairs  # noqa: E402
from geco.anomaly import detect_anomalies  # noqa: E402
from geco.features import load_dinov2  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ytf-data", required=True)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg-values", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--no-crop", action="store_true")
    args = parser.parse_args()
    crop = not args.no_crop

    print("Sampling pairs once (reused across every reg value)...")
    normal_pairs = sample_cross_video_pairs(args.ytf_data, n_pairs=args.n_pairs, seed=args.seed, crop_to_face=crop, verbose=False)
    anomalous_pairs = sample_cross_person_pairs(args.ytf_data, n_pairs=args.n_pairs, seed=args.seed, crop_to_face=crop, verbose=False)
    print(f"  {len(normal_pairs)} same-person, {len(anomalous_pairs)} different-person pairs")

    print("Loading DINOv2...")
    model = load_dinov2()

    print(f"\n{'reg':<10}{'converged?':<14}{'AUROC':<10}")
    print("-" * 34)
    for reg in args.reg_values:
        labels, scores = [], []
        converged = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for test_img, ref_img in normal_pairs:
                result = detect_anomalies(model, test_img, ref_img, match_kwargs={"reg": reg})
                scores.append(result.anomaly_score)
                labels.append(0)
            for test_img, ref_img in anomalous_pairs:
                result = detect_anomalies(model, test_img, ref_img, match_kwargs={"reg": reg})
                scores.append(result.anomaly_score)
                labels.append(1)
            if any("did not converge" in str(w.message) for w in caught):
                converged = False

        auroc = roc_auc_score(labels, scores)
        print(f"{reg:<10}{'yes' if converged else 'NO':<14}{auroc:<10.4f}")

    print("\nPick the reg with the best AUROC (ties broken toward the one that converges) "
          "and pass it as geco_match's default for face pairs, or thread it through "
          "evaluate_face_verification_anomaly.py's match_kwargs.")


if __name__ == "__main__":
    main()
