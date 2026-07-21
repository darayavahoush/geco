"""Translate a clicked pixel on the source image into a matched pixel on the target image."""

from __future__ import annotations

from dataclasses import dataclass

from .features import GRID_SIZE, IMG_SIZE, PATCH_SIZE
from .matching import MatchResult


@dataclass
class KeypointMatch:
    src_pixel: tuple[int, int]
    trg_pixel: tuple[int, int]
    src_patch: tuple[int, int]
    trg_patch: tuple[int, int]
    confidence: float  # transport-plan mass assigned to the best match
    is_dustbin: bool  # True if the model thinks there is no confident match


def transfer_keypoint(
    result: MatchResult, pixel_x: int, pixel_y: int, image_size: int = IMG_SIZE
) -> KeypointMatch:
    """Map a pixel coordinate (in `image_size`-scaled source image space) to the target image.

    Args:
        result: output of `geco_match`.
        pixel_x, pixel_y: clicked pixel coordinates, assumed relative to a
            square image resized to `image_size` (matching the frontend's
            displayed/resized image).
        image_size: side length the pixel coordinates are relative to.
    """
    grid = result.grid_size
    scale = grid * PATCH_SIZE / image_size  # convert to the model's native 518-space

    patch_x = min(int(pixel_x * scale) // PATCH_SIZE, grid - 1)
    patch_y = min(int(pixel_y * scale) // PATCH_SIZE, grid - 1)
    patch_idx = patch_y * grid + patch_x

    match_row = result.transport_plan[patch_idx, :]  # includes dustbin at index -1
    best_idx = match_row.argmax().item()
    dustbin_idx = match_row.numel() - 1
    is_dustbin = best_idx == dustbin_idx

    if is_dustbin:
        trg_patch_x, trg_patch_y = patch_x, patch_y  # fall back, flagged as low-confidence
    else:
        trg_patch_x = best_idx % grid
        trg_patch_y = best_idx // grid

    # Report the pixel at the center of the matched patch, scaled back to image_size.
    px = int((trg_patch_x * PATCH_SIZE + PATCH_SIZE / 2) / scale)
    py = int((trg_patch_y * PATCH_SIZE + PATCH_SIZE / 2) / scale)

    return KeypointMatch(
        src_pixel=(pixel_x, pixel_y),
        trg_pixel=(px, py),
        src_patch=(patch_x, patch_y),
        trg_patch=(trg_patch_x, trg_patch_y),
        confidence=match_row.max().item(),
        is_dustbin=is_dustbin,
    )
