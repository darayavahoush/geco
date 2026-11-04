"""Evaluate a trained face embedding on the standard VERIFICATION-PAIRS protocol
(same format LFW and most face benchmarks use): a list of (image_a, image_b,
is_same_person) triples. Reports accuracy at a swept range of thresholds and
picks the best one -- this is also how you CALIBRATE the threshold used by
geco.face.verify() before trusting it on new photos.

Pairs file format (plain text, one pair per line):
    path/to/photo_a.jpg  path/to/photo_b.jpg  1
    path/to/photo_c.jpg  path/to/photo_d.jpg  0

(1 = same person, 0 = different person -- build this yourself from your
dataset, or adapt LFW's pairs.txt format if using that benchmark directly.)

Usage:
    python scripts/eval/evaluate_face_verification.py \
        --pairs-file ./datasets/faces/pairs.txt \
        --head-weights ./face_head.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from PIL import Image

from geco.face import FaceEmbedder, FaceProjectionHead  # noqa: E402


def load_pairs(pairs_file: Path) -> list[tuple[str, str, bool]]:
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            path_a, path_b, label = parts
            pairs.append((path_a, path_b, label == "1"))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--head-weights", default=None, help="Path to trained head .pt file (omit to use an UNTRAINED head, as a baseline)")
    parser.add_argument("--write-calibration", action="store_true", help="Save threshold + stats next to the head weights")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pairs_file))
    if not pairs:
        raise SystemExit(f"No valid pairs found in {args.pairs_file}")
    print(f"Loaded {len(pairs)} verification pairs.")

    print("Loading DINOv2 backbone...")
    head = FaceProjectionHead()
    if args.head_weights:
        head.load_state_dict(torch.load(args.head_weights, map_location="cpu"))
        print(f"Loaded trained head from {args.head_weights}")
    else:
        print("WARNING: using an UNTRAINED head -- this is a baseline only, expect weak results.")
    embedder = FaceEmbedder(head=head)

    similarities = []
    labels = []
    for i, (path_a, path_b, is_same) in enumerate(pairs):
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        emb_a = embedder.embed(img_a)
        emb_b = embedder.embed(img_b)
        sim = torch.dot(emb_a, emb_b).item()
        similarities.append(sim)
        labels.append(is_same)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs scored")

    import numpy as np

    from geco.calibration import CALIBRATION_FILENAME, calibrate, save_calibration

    report = calibrate(np.array(similarities), np.array(labels))
    print(f"\nScored {len(pairs)} pairs.\n")
    print(f"same-person  cosine: mean {report['same_mean']:.3f} +/- {report['same_std']:.3f}")
    print(f"different    cosine: mean {report['diff_mean']:.3f} +/- {report['diff_std']:.3f}")
    print(f"AUC {report['auc']:.4f}   EER {report['eer']:.4f}")
    print(f"threshold {report['threshold']:.3f}  (in-sample accuracy {report['in_sample_accuracy']:.4f})")
    print(
        f"held-out check: threshold {report['heldout_threshold']:.3f} fit on half the pairs "
        f"-> accuracy {report['heldout_accuracy']:.4f} on the other half  <-- quote THIS number"
    )
    if abs(report["diff_mean"] - report["same_mean"]) < 2 * report["margin_scale"]:
        print("WARNING: class means are within 2 std of each other; the embedding separates identities weakly.")
    if report["diff_mean"] > 0.8:
        print("WARNING: different-person pairs score > 0.8 on average -- embeddings are crowded into a narrow cone.")

    out = Path(args.head_weights).with_name(CALIBRATION_FILENAME) if args.head_weights else Path(CALIBRATION_FILENAME)
    if args.write_calibration:
        save_calibration(report, out)
        print(f"Wrote calibration to {out}")
    else:
        print(f"(pass --write-calibration to save to {out}; the backend reads it at startup)")


if __name__ == "__main__":
    main()
