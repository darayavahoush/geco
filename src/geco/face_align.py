"""Face alignment via 2-point (eye) landmarks -- the same category of preprocessing
used by ArcFace/FaceNet-style pipelines: rotate so the eyes are level, scale so
inter-ocular distance matches a fixed reference, and crop to a canonical square. This
removes pose/tilt variance the network would otherwise have to be invariant to,
freeing its capacity for actual identity signal rather than pose robustness.

Two entry points:
  align_face(PIL.Image)                          -- runs MediaPipe eye detection first
  align_from_points(rgb_array, left_eye, right_eye) -- given ALREADY-KNOWN eye positions
                                                        (e.g. a dataset''s own ground-truth
                                                        landmarks, which is more reliable
                                                        than re-running detection)

cv2 and mediapipe are imported in SEPARATE try/except blocks -- if mediapipe is missing,
cv2 (needed by align_from_points, which doesn''t use mediapipe at all) stays available.
Bundling them in one try/except was a real bug this project hit: a missing mediapipe
silently disabled cv2 too, making every alignment call fall back to a plain resize.

Falls back to a plain center-resize if no face is detected, dependencies are missing,
or the implied zoom scale looks implausible (protects against bad/corrupt landmarks
producing a garbled crop).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    import mediapipe as mp

    _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
    )
    _MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    _mp_face_mesh = None
    _MEDIAPIPE_AVAILABLE = False

_ALIGN_AVAILABLE = _CV2_AVAILABLE and _MEDIAPIPE_AVAILABLE  # align_face() needs both

ALIGN_SIZE = 224  # output crop size; fed into preprocess_image()''s own resize afterward
_DETECTION_MIN_DIM = 400  # upscale to at least this on the longer side before detecting

_LEFT_EYE_IDX = [33, 133]
_RIGHT_EYE_IDX = [362, 263]

_REF_LEFT_EYE = (0.342 * ALIGN_SIZE, 0.461 * ALIGN_SIZE)
_REF_RIGHT_EYE = (0.656 * ALIGN_SIZE, 0.460 * ALIGN_SIZE)
_REF_INTEROCULAR = _REF_RIGHT_EYE[0] - _REF_LEFT_EYE[0]

_MIN_SCALE = 0.3
_MAX_SCALE = 3.0


def _fallback(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((ALIGN_SIZE, ALIGN_SIZE))


def align_from_points(rgb_array: np.ndarray, left_eye: tuple, right_eye: tuple):
    """Core alignment transform given ALREADY-KNOWN eye positions in rgb_array''s own
    pixel coordinate space. Returns an aligned ALIGN_SIZE x ALIGN_SIZE RGB uint8
    numpy array, or None if the given points look implausible or cv2 is unavailable.
    """
    if not _CV2_AVAILABLE:
        return None

    detected_interocular = ((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2) ** 0.5
    if detected_interocular < 1e-3:
        return None

    implied_scale = _REF_INTEROCULAR / detected_interocular
    if not (_MIN_SCALE <= implied_scale <= _MAX_SCALE):
        return None

    src_pts = np.array([left_eye, right_eye], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([_REF_LEFT_EYE, _REF_RIGHT_EYE], dtype=np.float32).reshape(-1, 1, 2)

    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if matrix is None:
        return None

    return cv2.warpAffine(
        rgb_array, matrix, (ALIGN_SIZE, ALIGN_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def align_face(img: Image.Image) -> Image.Image:
    """Returns a rotation/scale/translation-aligned face crop, ALIGN_SIZE x ALIGN_SIZE,
    using MediaPipe eye-landmark detection.
    """
    if not _ALIGN_AVAILABLE:
        return _fallback(img)

    rgb_orig = np.array(img.convert("RGB"))
    h_orig, w_orig = rgb_orig.shape[:2]

    longer_side = max(w_orig, h_orig)
    if longer_side < _DETECTION_MIN_DIM:
        upscale_factor = _DETECTION_MIN_DIM / longer_side
        detect_w, detect_h = int(w_orig * upscale_factor), int(h_orig * upscale_factor)
        rgb_detect = cv2.resize(rgb_orig, (detect_w, detect_h), interpolation=cv2.INTER_CUBIC)
    else:
        rgb_detect = rgb_orig

    result = _mp_face_mesh.process(rgb_detect)
    if not result.multi_face_landmarks:
        return _fallback(img)

    landmarks = result.multi_face_landmarks[0].landmark

    def _avg_point(idxs):
        xs = [landmarks[i].x * w_orig for i in idxs]
        ys = [landmarks[i].y * h_orig for i in idxs]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    left_eye = _avg_point(_LEFT_EYE_IDX)
    right_eye = _avg_point(_RIGHT_EYE_IDX)

    aligned = align_from_points(rgb_orig, left_eye, right_eye)
    if aligned is None:
        return _fallback(img)
    return Image.fromarray(aligned)
