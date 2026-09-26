"""
Loader for selfishgene/youtube-faces-with-facial-keypoints (Kaggle), against the
CONFIRMED real schema (verified 2026-09-26):

  Per-video file: face_data/ytf/youtube_faces_with_keypoints_full_<N>/
                    youtube_faces_with_keypoints_full_<N>/<videoID>.npz
    colorImages : (H, W, 3, n_frames)  uint8
    landmarks2D : (68, 2, n_frames)    float64
    landmarks3D : (68, 3, n_frames)    float64
    boundingBox : (4, 2, n_frames)     float64

  Index csv: youtube_faces_with_keypoints_full.csv
    columns: videoID, personName, imageHeight, imageWidth, videoDuration,
             averageFaceSize, numVideosForPerson
    (no shard/split column -- a video's shard isn't known from the csv,
     so we locate .npz files by filename across all four shards)
"""
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 36, 45  # standard 68-point convention


@lru_cache(maxsize=1)
def build_video_index(data_dir: str) -> dict:
    """videoID -> absolute .npz path, built once by scanning all shards."""
    paths = glob.glob(f"{data_dir}/**/*.npz", recursive=True)
    return {Path(p).stem: p for p in paths}


def _resolve(data_dir: str, video_id: str) -> str:
    index = build_video_index(data_dir)
    if video_id not in index:
        raise FileNotFoundError(f"{video_id}.npz not found under {data_dir} (scanned {len(index)} files)")
    return index[video_id]


def load_video(data_dir: str, video_id: str):
    """Returns (frames: list of HxWx3 uint8, landmarks2d: (n_frames,68,2)).

    Loads EVERY frame -- use only when you actually need the whole clip
    (e.g. temporal/integrity work). For sampling a single random frame from
    a video, use load_single_frame() instead: some YTF clips run to 6000+
    frames, and materializing all of them just to keep one is the reason
    the pair-samplers below used to stall.
    """
    d = np.load(_resolve(data_dir, video_id), allow_pickle=True)
    n_frames = d["colorImages"].shape[-1]
    frames = [d["colorImages"][..., t] for t in range(n_frames)]
    landmarks2d = np.moveaxis(d["landmarks2D"], -1, 0)  # (68,2,n_frames) -> (n_frames,68,2)
    return frames, landmarks2d


def load_single_frame(data_dir: str, video_id: str, rng: np.random.Generator, crop_to_face: bool = True, pad_frac: float = 0.4):
    """Returns one random frame (HxWx3 uint8) from a video WITHOUT loading the rest.
    O(1) in frame count -- this is what the pair-samplers should use.

    crop_to_face: crop to the frame's boundingBox, padded by pad_frac on each
    side (default 0.4 = 40% of box width/height added per side). A tight crop
    to the raw box measured WORSE than no crop at all in testing (likely
    cutting off jaw/hair/ear context DINOv2 patches were actually using) --
    padding is a middle ground between "all the background" and "too tight."
    Confirmed box format (verified 2026-09-26): 4 corner (x,y) points, e.g.
    [[68,85],[68,173],[156,85],[156,173]] -- min/max across them is correct.
    """
    d = np.load(_resolve(data_dir, video_id), allow_pickle=True)
    n_frames = d["colorImages"].shape[-1]
    t = int(rng.integers(0, n_frames))
    frame = d["colorImages"][..., t]
    if not crop_to_face:
        return frame

    box = d["boundingBox"][..., t]
    x0, y0 = box[:, 0].min(), box[:, 1].min()
    x1, y1 = box[:, 0].max(), box[:, 1].max()
    bw, bh = x1 - x0, y1 - y0
    x0, y0 = x0 - pad_frac * bw, y0 - pad_frac * bh
    x1, y1 = x1 + pad_frac * bw, y1 + pad_frac * bh
    h, w = frame.shape[:2]
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1]


def iter_ytf_clips(data_dir, limit=None, min_frames=10):
    """
    Yields (frames, seed_points, seed_labels) per clip, for use by future
    temporal/integrity work -- one clip per video in the index csv. Uses
    load_video() (full clip) deliberately, since temporal work needs every frame.
    """
    csv_path = next(Path(data_dir).glob("*.csv"))
    df = pd.read_csv(csv_path)

    yielded = 0
    for video_id in df["videoID"]:
        try:
            frames, landmarks2d = load_video(data_dir, video_id)
        except FileNotFoundError:
            continue  # shard not downloaded / row without matching npz
        if len(frames) < min_frames:
            continue
        yield frames, landmarks2d[0], list(range(68))  # seed from first frame's 68 landmarks
        yielded += 1
        if limit and yielded >= limit:
            return


def sample_cross_video_pairs(data_dir, n_pairs=2000, seed=0, verbose=True, crop_to_face=True):
    """
    SAME-person pairs (two different videos, same identity, pose/expression varies).
    "Normal" class for face-verification-as-anomaly-detection: correspondence
    SHOULD hold, anomaly score should be LOW.

    Returns list of (img1_pil, img2_pil) PIL RGB images -- ready for geco.anomaly.detect_anomalies.
    """
    from PIL import Image
    csv_path = next(Path(data_dir).glob("*.csv"))
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)

    pairs = []
    for person, group in df.groupby("personName"):
        video_ids = group["videoID"].unique()
        if len(video_ids) < 2:
            continue
        v1, v2 = rng.choice(video_ids, size=2, replace=False)
        try:
            f1 = load_single_frame(data_dir, v1, rng, crop_to_face=crop_to_face)
            f2 = load_single_frame(data_dir, v2, rng, crop_to_face=crop_to_face)
        except FileNotFoundError:
            continue
        pairs.append((Image.fromarray(f1), Image.fromarray(f2)))
        if verbose and len(pairs) % 10 == 0:
            print(f"    ...{len(pairs)}/{n_pairs} same-person pairs sampled")
        if len(pairs) >= n_pairs:
            break
    return pairs


def sample_cross_person_pairs(data_dir, n_pairs=2000, seed=0, verbose=True, crop_to_face=True):
    """
    DIFFERENT-person pairs. "Anomalous" class: correspondence SHOULD break,
    anomaly score should be HIGH. Mirrors sample_anomalous_pairs() in
    evaluate_component2b.py, drawing from YTF identities instead of CUB classes.

    Returns list of (img1_pil, img2_pil).
    """
    from PIL import Image
    csv_path = next(Path(data_dir).glob("*.csv"))
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    people = df["personName"].unique()

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 5:
        attempts += 1
        p1, p2 = rng.choice(people, size=2, replace=False)
        v1 = rng.choice(df[df.personName == p1]["videoID"].values)
        v2 = rng.choice(df[df.personName == p2]["videoID"].values)
        try:
            f1 = load_single_frame(data_dir, v1, rng, crop_to_face=crop_to_face)
            f2 = load_single_frame(data_dir, v2, rng, crop_to_face=crop_to_face)
        except FileNotFoundError:
            continue
        pairs.append((Image.fromarray(f1), Image.fromarray(f2)))
        if verbose and len(pairs) % 10 == 0:
            print(f"    ...{len(pairs)}/{n_pairs} different-person pairs sampled ({attempts} attempts)")
    return pairs
