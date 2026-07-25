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
    confidence: float  # fraction of this patch's OWN transport mass that went to its best match (0-1)
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

    # Raw transport-plan mass in a row scales as ~1/N (probability spread over N patches +
    # dustbin), so it's meaningless as an absolute "confidence" — a perfect match on a
    # 1369-patch grid would still only carry ~0.0007 raw mass. Normalize by the row's own
    # total so confidence means "how much of THIS patch's assigned mass is concentrated on
    # its best match" — a proper 0-1 score independent of grid size.
    row_sum = match_row.sum().item()
    normalized_confidence = match_row.max().item() / (row_sum + 1e-8)

    return KeypointMatch(
        src_pixel=(pixel_x, pixel_y),
        trg_pixel=(px, py),
        src_patch=(patch_x, patch_y),
        trg_patch=(trg_patch_x, trg_patch_y),
        confidence=normalized_confidence,
        is_dustbin=is_dustbin,
    )
