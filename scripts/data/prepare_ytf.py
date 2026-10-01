"""Extracts aligned face-frame JPGs from the Kaggle "YouTube Faces With Facial
Keypoints" .npz shards, using the dataset''s OWN provided 68-point landmarks for
alignment (more reliable than re-running MediaPipe on these frames, since the dataset
already ships accurate per-frame landmark annotations -- see geco.face_align.align_from_points).

Writes one-identity-per-folder, matching the LFW layout already used elsewhere in this
project, so train_face_arcface.py needs no changes beyond accepting multiple dataset roots.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geco.face_align import align_from_points  # noqa: E402

# Standard 68-point (iBUG/dlib) landmark scheme: indices 36-41 are one eye, 42-47 the
# other. In a forward-facing photo these sit on the image-left and image-right
# respectively, matching this project''s _REF_LEFT_EYE/_REF_RIGHT_EYE convention.
_LEFT_EYE_LANDMARKS = list(range(36, 42))
_RIGHT_EYE_LANDMARKS = list(range(42, 48))


def find_npz_shards(ytf_root: Path) -> dict[str, Path]:
    """Maps videoID -> its .npz file path, across all shard folders (Kaggle''s zip
    nests a duplicate-named subfolder inside each numbered shard folder).
    """
    mapping = {}
    for shard_dir in sorted(ytf_root.glob("youtube_faces_with_keypoints_full_*")):
        if not shard_dir.is_dir():
            continue
        inner = shard_dir / shard_dir.name
        search_dir = inner if inner.exists() else shard_dir
        for npz_path in search_dir.glob("*.npz"):
            mapping[npz_path.stem] = npz_path
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ytf-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-identities", type=int, default=500)
    parser.add_argument("--max-frames-per-video", type=int, default=8)
    parser.add_argument(
        "--id-prefix", default="ytf_",
        help="Prefix for identity folder names, to avoid accidentally merging with an "
        "LFW folder of the same name that might be a different real person.",
    )
    args = parser.parse_args()

    csv_path = args.ytf_root / "youtube_faces_with_keypoints_full.csv"
    rows = list(csv.DictReader(csv_path.open()))
    print(f"Loaded {len(rows)} video entries from CSV.")

    videos_by_person: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        videos_by_person[row["personName"]].append(row["videoID"])

    chosen_people = sorted(videos_by_person.keys())[: args.max_identities]
    print(f"Using {len(chosen_people)} identities (out of {len(videos_by_person)} available).")

    npz_index = find_npz_shards(args.ytf_root)
    print(f"Indexed {len(npz_index)} .npz video files across shard folders.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for person_i, person in enumerate(chosen_people, start=1):
        out_dir = args.output_root / f"{args.id_prefix}{person}"
        out_dir.mkdir(exist_ok=True)

        for video_id in videos_by_person[person]:
            npz_path = npz_index.get(video_id)
            if npz_path is None:
                continue

            data = np.load(npz_path)
            color_images = data["colorImages"]  # [H, W, 3, num_frames]
            landmarks2d = data["landmarks2D"]  # [68, 2, num_frames]
            num_frames = color_images.shape[-1]

            if num_frames <= args.max_frames_per_video:
                frame_indices = range(num_frames)
            else:
                step = num_frames / args.max_frames_per_video
                frame_indices = [int(i * step) for i in range(args.max_frames_per_video)]

            for frame_idx in frame_indices:
                frame = color_images[:, :, :, frame_idx]  # [H, W, 3] uint8
                lm = landmarks2d[:, :, frame_idx]  # [68, 2]

                left_eye = tuple(lm[_LEFT_EYE_LANDMARKS].mean(axis=0))
                right_eye = tuple(lm[_RIGHT_EYE_LANDMARKS].mean(axis=0))

                aligned = align_from_points(frame, left_eye, right_eye)
                if aligned is None:
                    continue

                out_path = out_dir / f"{video_id}_{frame_idx}.jpg"
                Image.fromarray(aligned).save(out_path)
                total_saved += 1

        if person_i % 50 == 0:
            print(f"  {person_i}/{len(chosen_people)} identities processed, {total_saved} frames saved so far")

    print(f"Done. Saved {total_saved} aligned frames across {len(chosen_people)} identities to {args.output_root}")


if __name__ == "__main__":
    main()
