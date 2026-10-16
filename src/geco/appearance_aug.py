"""Synthetic appearance changes (glasses, beard) for ALIGNED 224x224 face crops.

Why this exists: the face head is trained on photos where most people look the same
from image to image. Add or remove glasses/a beard and the embedding drifts, so the
same person scores like a stranger. Training the head on clean + glasses + beard views
of the SAME identity (all sharing one label) teaches it to ignore those regions.

Everything here works in the canonical aligned frame produced by geco.face_align
(eyes at fixed positions, 224x224), so shapes can be placed by fixed fractions of the
crop instead of needing a landmark model. Draw -> blend -> return a new PIL image.

These are deliberately simple renders, not photoreal. They are a training/eval aid;
real photos of the same person with/without glasses or beard (e.g. CelebA attributes,
or your own enrolment photos) remain the real test -- see
scripts/eval/evaluate_appearance_robustness.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python(-headless) is required for geco.appearance_aug") from exc

from .face_align import ALIGN_SIZE, _REF_LEFT_EYE, _REF_RIGHT_EYE, _REF_INTEROCULAR

S = ALIGN_SIZE

_FRAME_COLORS = [
    (15, 15, 15), (40, 30, 25), (90, 60, 35), (150, 150, 155), (170, 140, 60),
    (120, 20, 30), (25, 40, 110), (200, 200, 205),
]
_HAIR_COLORS = [
    (20, 15, 12), (45, 30, 20), (75, 50, 30), (110, 70, 40), (150, 90, 45),
    (120, 120, 120), (180, 180, 180),
]


def _rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _to_array(img: Image.Image) -> np.ndarray:
    if img.size != (S, S):
        img = img.resize((S, S), Image.Resampling.LANCZOS)
    return np.array(img.convert("RGB"))


def add_glasses(img: Image.Image, rng=None, sunglasses: bool | None = None) -> Image.Image:
    """Draw a random pair of glasses over the eyes of an aligned crop."""
    r = _rng(rng)
    arr = _to_array(img).astype(np.float32)
    layer = np.zeros_like(arr)
    alpha = np.zeros(arr.shape[:2], np.float32)

    left = np.array(_REF_LEFT_EYE, np.float32) + r.normal(0, 1.5, 2)
    right = np.array(_REF_RIGHT_EYE, np.float32) + r.normal(0, 1.5, 2)
    iod = _REF_INTEROCULAR
    sun = bool(r.random() < 0.3) if sunglasses is None else sunglasses
    shape = r.choice(["round", "rect", "aviator"])
    lens_w = iod * r.uniform(0.78, 0.98)
    lens_h = lens_w * {"round": r.uniform(0.85, 1.0), "rect": r.uniform(0.60, 0.75), "aviator": r.uniform(0.80, 0.95)}[shape]
    thick = int(r.integers(2, 6))
    color = _FRAME_COLORS[int(r.integers(len(_FRAME_COLORS)))]
    tilt = float(r.normal(0, 2.5))

    lens_mask = np.zeros(arr.shape[:2], np.uint8)
    frame_mask = np.zeros(arr.shape[:2], np.uint8)

    for c in (left, right):
        center = (int(c[0]), int(c[1] + (3 if shape == "aviator" else 0)))
        axes = (int(lens_w / 2), int(lens_h / 2))
        if shape == "rect":
            pts = cv2.ellipse2Poly(center, axes, int(tilt), 0, 360, 20)
            # squarer: push ellipse points outward along each axis
            cx, cy = center
            pts = np.array([[cx + np.sign(x - cx) * abs(x - cx) ** 0.8 * axes[0] ** 0.2,
                             cy + np.sign(y - cy) * abs(y - cy) ** 0.8 * axes[1] ** 0.2] for x, y in pts], np.int32)
            cv2.fillPoly(lens_mask, [pts], 255)
            cv2.polylines(frame_mask, [pts], True, 255, thick, cv2.LINE_AA)
        else:
            cv2.ellipse(lens_mask, center, axes, tilt, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(frame_mask, center, axes, tilt, 0, 360, 255, thick, cv2.LINE_AA)

    # bridge + temples
    by = int((left[1] + right[1]) / 2 - lens_h * 0.1)
    cv2.line(frame_mask, (int(left[0] + lens_w / 2), by), (int(right[0] - lens_w / 2), by), 255, max(2, thick - 1), cv2.LINE_AA)
    cv2.line(frame_mask, (int(left[0] - lens_w / 2), int(left[1] - 2)), (0, int(left[1] - 6)), 255, max(2, thick - 1), cv2.LINE_AA)
    cv2.line(frame_mask, (int(right[0] + lens_w / 2), int(right[1] - 2)), (S - 1, int(right[1] - 6)), 255, max(2, thick - 1), cv2.LINE_AA)

    if sun:
        tint = np.array(color, np.float32) * 0.35
        lens_alpha = r.uniform(0.75, 0.95)
    else:
        tint = np.array([200, 215, 230], np.float32)  # faint glass glare
        lens_alpha = r.uniform(0.05, 0.18)

    lens_a = cv2.GaussianBlur(lens_mask.astype(np.float32) / 255.0, (3, 3), 0) * lens_alpha
    frame_a = cv2.GaussianBlur(frame_mask.astype(np.float32) / 255.0, (3, 3), 0)
    out = arr * (1 - lens_a[..., None]) + tint[None, None, :] * lens_a[..., None]
    out = out * (1 - frame_a[..., None] * 0.97) + np.array(color, np.float32)[None, None, :] * frame_a[..., None] * 0.97
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _beard_polygon(r: np.random.Generator, style: str) -> np.ndarray:
    """Polygon (pixels) of the beard region in the canonical aligned frame."""
    j = lambda s=0.012: r.normal(0, s)  # noqa: E731
    if style == "stubble":
        pts = [(0.17, 0.58), (0.22, 0.76), (0.34, 0.92), (0.50, 0.985), (0.66, 0.92), (0.78, 0.76), (0.83, 0.58),
               (0.70, 0.64), (0.60, 0.67), (0.50, 0.66), (0.40, 0.67), (0.30, 0.64)]
    elif style == "goatee":
        pts = [(0.38, 0.80), (0.42, 0.74), (0.58, 0.74), (0.62, 0.80), (0.60, 0.92), (0.50, 0.985), (0.40, 0.92)]
    else:  # full
        pts = [(0.13, 0.50), (0.18, 0.74), (0.30, 0.91), (0.50, 1.0), (0.70, 0.91), (0.82, 0.74), (0.87, 0.50),
               (0.74, 0.60), (0.64, 0.665), (0.50, 0.65), (0.36, 0.665), (0.26, 0.60)]
    return np.array([[(x + j()) * S, (y + j()) * S] for x, y in pts], np.int32)


def add_beard(img: Image.Image, rng=None, style: str | None = None) -> Image.Image:
    """Overlay a textured beard (full / stubble / goatee, optional mustache) on an aligned crop."""
    r = _rng(rng)
    arr = _to_array(img).astype(np.float32)
    style = style or str(r.choice(["full", "full", "stubble", "goatee"]))
    color = np.array(_HAIR_COLORS[int(r.integers(len(_HAIR_COLORS)))], np.float32)

    mask = np.zeros((S, S), np.uint8)
    cv2.fillPoly(mask, [_beard_polygon(r, style)], 255)
    if style != "stubble" and r.random() < 0.8:  # mustache above the mouth
        mx, my = S * 0.5, S * 0.715
        pts = cv2.ellipse2Poly((int(mx), int(my)), (int(S * r.uniform(0.11, 0.16)), int(S * 0.04)), 0, 180, 360, 15)
        cv2.fillPoly(mask, [pts], 255)
    # keep the lips visible
    cv2.ellipse(mask, (int(S * 0.5), int(S * 0.772)), (int(S * 0.095), int(S * 0.030)), 0, 0, 360, 0, -1, cv2.LINE_AA)

    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), r.uniform(2.0, 4.0))
    density = {"stubble": r.uniform(0.35, 0.55), "full": r.uniform(0.80, 0.95), "goatee": r.uniform(0.75, 0.92)}[style]

    # hair texture: base colour + per-pixel noise + short strands
    noise = r.normal(0, 14, (S, S, 1)).astype(np.float32)
    tex = np.clip(color[None, None, :] + noise, 0, 255)
    strands = np.zeros((S, S), np.float32)
    for _ in range(900 if style != "stubble" else 500):
        x, y = r.integers(0, S, 2)
        ang = np.deg2rad(r.normal(90, 25))
        l = r.integers(3, 9)
        cv2.line(strands, (int(x), int(y)), (int(x + l * np.cos(ang)), int(y + l * np.sin(ang))), 1.0, 1, cv2.LINE_AA)
    tex = tex * (1 - 0.35 * strands[..., None])

    a = (soft * density)[..., None]
    out = arr * (1 - a) + tex * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


APPEARANCE_MODES = ("glasses", "beard", "both")


def random_appearance_change(img: Image.Image, rng=None, mode: str | None = None) -> tuple[Image.Image, str]:
    """Apply a random appearance change. Returns (new_image, tag in APPEARANCE_MODES)."""
    r = _rng(rng)
    mode = mode or str(r.choice(APPEARANCE_MODES))
    out = img
    if mode in ("glasses", "both"):
        out = add_glasses(out, r)
    if mode in ("beard", "both"):
        out = add_beard(out, r)
    return out, mode
