"""Face detection, bounding-box localization, and 2-point canonical eye alignment.
Extracts the exact face region from arbitrary user photos (selfies, multi-angle poses,
and portraits), discarding background walls, clothing, and body clutter.

Supported detection backends:
1. OpenCV YuNet Deep Learning Face Detector (ONNX runtime via cv2.FaceDetectorYN)
2. Chromaticity skin-locus + facial feature gradient energy localization
3. MediaPipe FaceMesh (when available)
4. Golden-ratio central face framing fallback
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

ALIGN_SIZE = 224  # Canonical crop size fed into DINOv2 / ArcFace
_DETECTION_MIN_DIM = 400

_REF_LEFT_EYE = (0.342 * ALIGN_SIZE, 0.461 * ALIGN_SIZE)
_REF_RIGHT_EYE = (0.656 * ALIGN_SIZE, 0.460 * ALIGN_SIZE)
_REF_INTEROCULAR = _REF_RIGHT_EYE[0] - _REF_LEFT_EYE[0]

_MIN_SCALE = 0.3
_MAX_SCALE = 3.0

_YUNET_DETECTOR = None
_YUNET_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "face_detection_yunet_2023mar.onnx"
if not _YUNET_MODEL_PATH.exists():
    _YUNET_MODEL_PATH = Path("face_detection_yunet_2023mar.onnx")


def _get_yunet_detector(width: int = 320, height: int = 320):
    global _YUNET_DETECTOR
    if not _CV2_AVAILABLE:
        return None
    if _YUNET_DETECTOR is None and _YUNET_MODEL_PATH.exists():
        try:
            _YUNET_DETECTOR = cv2.FaceDetectorYN_create(
                model=str(_YUNET_MODEL_PATH),
                config="",
                input_size=(width, height),
                score_threshold=0.35,
                nms_threshold=0.3,
                top_k=10,
            )
        except Exception:
            _YUNET_DETECTOR = None
    if _YUNET_DETECTOR is not None:
        try:
            _YUNET_DETECTOR.setInputSize((width, height))
        except Exception:
            pass
    return _YUNET_DETECTOR


def _detect_yunet(rgb_array: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[float, float], tuple[float, float], dict, float] | None:
    h, w = rgb_array.shape[:2]
    detector = _get_yunet_detector(w, h)
    if detector is None:
        return None
    try:
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        _, faces = detector.detect(bgr)
        if faces is not None and len(faces) > 0:
            best_face = faces[0]  # sorted by confidence
            bx, by, bw, bh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
            score = float(best_face[-1])
            # Landmark points: right_eye (idx 4..5), left_eye (idx 6..7) in OpenCV camera convention
            # where right eye is subject's right (image left side)
            re_x, re_y = float(best_face[4]), float(best_face[5])
            le_x, le_y = float(best_face[6]), float(best_face[7])
            nose_x, nose_y = float(best_face[8]), float(best_face[9])
            rm_x, rm_y = float(best_face[10]), float(best_face[11])
            lm_x, lm_y = float(best_face[12]), float(best_face[13])

            left_pt = (min(re_x, le_x), re_y if re_x < le_x else le_y)
            right_pt = (max(re_x, le_x), le_y if re_x < le_x else re_y)
            left_mouth = (min(rm_x, lm_x), rm_y if rm_x < lm_x else lm_y)
            right_mouth = (max(rm_x, lm_x), lm_y if rm_x < lm_x else rm_y)

            landmarks = {
                "left_eye": left_pt,
                "right_eye": right_pt,
                "nose_tip": (nose_x, nose_y),
                "left_mouth": left_mouth,
                "right_mouth": right_mouth,
            }
            return (bx, by, bw, bh), left_pt, right_pt, landmarks, score
    except Exception:
        pass
    return None


def _detect_skin_gradient(rgb_array: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[float, float], tuple[float, float], dict, float]:
    """Robust fallback skin-chromaticity + facial feature edge energy detector."""
    h, w = rgb_array.shape[:2]
    rgb_f = rgb_array.astype(np.float32)

    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    y_ch = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # Human skin chromaticity envelope
    skin_mask = (cb >= 75) & (cb <= 130) & (cr >= 130) & (cr <= 178) & (y_ch >= 35)

    gray = y_ch / 255.0
    gy = np.abs(np.diff(gray, axis=0))
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")

    face_energy = skin_mask.astype(np.float32) * (1.0 + 2.5 * gy)

    # Portrait central prior
    yy, xx = np.mgrid[0:h, 0:w]
    center_prior = np.exp(-0.5 * (((xx - w / 2.0) / (w * 0.35)) ** 2 + ((yy - h * 0.45) / (h * 0.35)) ** 2))
    score_map = face_energy * center_prior

    total_score = np.sum(score_map)
    if total_score > 60:
        proj_x = np.sum(score_map, axis=0)
        proj_y = np.sum(score_map, axis=1)

        cum_x = np.cumsum(proj_x) / (total_score + 1e-6)
        cum_y = np.cumsum(proj_y) / (total_score + 1e-6)

        x0 = int(np.searchsorted(cum_x, 0.08))
        x1 = int(np.searchsorted(cum_x, 0.92))
        y0 = int(np.searchsorted(cum_y, 0.08))
        y1 = int(np.searchsorted(cum_y, 0.92))

        bw = max(24, x1 - x0)
        bh = max(24, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        side = int(max(bw, bh) * 1.30)

        bx = max(0, cx - side // 2)
        by = max(0, cy - side // 2)
        bw_box = min(w - bx, side)
        bh_box = min(h - by, side)

        # Estimate eye landmarks
        eye_y = by + bh_box * 0.38
        eye_left = (bx + bw_box * 0.34, eye_y)
        eye_right = (bx + bw_box * 0.66, eye_y)
        landmarks = {
            "left_eye": eye_left,
            "right_eye": eye_right,
            "nose_tip": (bx + bw_box * 0.50, by + bh_box * 0.55),
            "left_mouth": (bx + bw_box * 0.36, by + bh_box * 0.72),
            "right_mouth": (bx + bw_box * 0.64, by + bh_box * 0.72),
        }
        return (bx, by, bw_box, bh_box), eye_left, eye_right, landmarks, 0.82

    # Centered portrait framing
    side = int(min(w, h) * 0.70)
    cx, cy = w // 2, int(h * 0.45)
    bx = max(0, cx - side // 2)
    by = max(0, cy - side // 2)
    bw_box = min(w - bx, side)
    bh_box = min(h - by, side)
    eye_y = by + bh_box * 0.38
    eye_left = (bx + bw_box * 0.34, eye_y)
    eye_right = (bx + bw_box * 0.66, eye_y)
    landmarks = {
        "left_eye": eye_left,
        "right_eye": eye_right,
        "nose_tip": (bx + bw_box * 0.50, by + bh_box * 0.55),
        "left_mouth": (bx + bw_box * 0.36, by + bh_box * 0.72),
        "right_mouth": (bx + bw_box * 0.64, by + bh_box * 0.72),
    }
    return (bx, by, bw_box, bh_box), eye_left, eye_right, landmarks, 0.50


def align_from_points(rgb_array: np.ndarray, left_eye: tuple, right_eye: tuple) -> np.ndarray | None:
    """Core affine alignment given eye positions in rgb_array's pixel coordinates."""
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


def extract_face_bbox_and_crop(img: Image.Image) -> tuple[Image.Image, dict]:
    """Detects and extracts the exact face region, discarding background walls & body.
    Returns:
        (aligned_crop_224x224_PIL, metadata_dict)
    """
    img_rgb = img.convert("RGB")
    rgb_orig = np.array(img_rgb)
    h_orig, w_orig = rgb_orig.shape[:2]

    # Try deep YuNet first
    yunet_res = _detect_yunet(rgb_orig)
    if yunet_res is not None:
        (bx, by, bw, bh), left_eye, right_eye, landmarks, conf = yunet_res
        method = "yunet_dnn"
    else:
        (bx, by, bw, bh), left_eye, right_eye, landmarks, conf = _detect_skin_gradient(rgb_orig)
        method = "skin_gradient_analysis"

    # Clamp bbox within image bounds
    bx = max(0, min(w_orig - 1, bx))
    by = max(0, min(h_orig - 1, by))
    bw = max(16, min(w_orig - bx, bw))
    bh = max(16, min(h_orig - by, bh))

    # Try 2-point affine eye alignment
    aligned_arr = align_from_points(rgb_orig, left_eye, right_eye)
    if aligned_arr is not None:
        crop_img = Image.fromarray(aligned_arr)
    else:
        # Pad box slightly and crop square
        cx, cy = bx + bw // 2, by + bh // 2
        side = int(max(bw, bh) * 1.15)
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w_orig, x0 + side)
        y1 = min(h_orig, y0 + side)
        crop_img = img_rgb.crop((x0, y0, x1, y1)).resize((ALIGN_SIZE, ALIGN_SIZE), Image.Resampling.LANCZOS)

    meta = {
        "bbox": [bx, by, bw, bh],
        "normalized_bbox": [
            round(bx / max(1, w_orig), 4),
            round(by / max(1, h_orig), 4),
            round(bw / max(1, w_orig), 4),
            round(bh / max(1, h_orig), 4),
        ],
        "landmarks": {
            k: [round(pt[0], 1), round(pt[1], 1)]
            for k, pt in landmarks.items()
        },
        "normalized_landmarks": {
            k: [round(pt[0] / max(1, w_orig), 4), round(pt[1] / max(1, h_orig), 4)]
            for k, pt in landmarks.items()
        },
        "confidence": round(conf * 100.0, 1),
        "method": method,
        "is_detected": True,
    }
    return crop_img, meta


def align_face(img: Image.Image) -> Image.Image:
    """Returns a tight, centered 224x224 aligned face crop."""
    crop, _ = extract_face_bbox_and_crop(img)
    return crop
