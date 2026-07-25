"""Component 2 / Track A — Sparse-to-Dense Annotation Propagation.

Given a small set of manually keypoint-annotated "seed" images, propagate
those keypoint labels onto an unlabeled target image using Component 1's
correspondence engine, without touching its internals.

Three independent quality signals are combined, each catching a different
failure mode:

1. **OT confidence** — how much transport mass the match got. Catches
   generically weak matches.
2. **Cycle consistency** — target -> seed -> does it land back near the
   original point? Catches drift and near-duplicate-texture confusion.
3. **Geometric consistency (RANSAC affine)** — for each seed, fit one affine
   transform across ALL of that seed's keypoints jointly, then flag any
   single point whose match disagrees with the rest of that seed's layout.
   This catches a failure mode neither of the above two catches: a
   confident, cycle-consistent match to the *wrong* semantic part (e.g.
   matching "left eye" to the target's right eye on a symmetric face) —
   which is locally plausible but globally inconsistent with where the
   other keypoints from the same seed landed.

A propagated point is accepted only if it clears the confidence and
cycle-consistency thresholds; the geometric check is used to prefer the
best candidate across seeds during voting, and reported as a diagnostic
flag either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .features import IMG_SIZE
from .geometry import fit_affine_ransac
from .keypoints import transfer_keypoint
from .matching import MatchResult, geco_match


@dataclass
class SeedAnnotation:
    """A labeled reference image: keypoint_name -> (x, y) in IMG_SIZE-scaled pixel space."""

    image: Image.Image
    keypoints: dict[str, tuple[int, int]]


@dataclass
class PropagatedKeypoint:
    name: str
    pixel: tuple[int, int]
    confidence: float  # OT transport-plan confidence
    cycle_error: float  # pixel distance between seed point and round-trip match; lower = better
    n_votes: int  # how many seeds contributed a candidate for this keypoint
    geometric_inlier: bool  # True if consistent with the rest of its seed's keypoint layout
    accepted: bool  # True if it passed confidence + cycle-consistency thresholds


def _cycle_consistency_error(
    model: torch.nn.Module,
    target_img: Image.Image,
    seed_img: Image.Image,
    target_pixel: tuple[int, int],
    seed_pixel: tuple[int, int],
    match_kwargs: dict,
) -> float:
    back_result = geco_match(model, target_img, seed_img, **match_kwargs)
    back_match = transfer_keypoint(back_result, target_pixel[0], target_pixel[1], image_size=IMG_SIZE)
    dx = back_match.trg_pixel[0] - seed_pixel[0]
    dy = back_match.trg_pixel[1] - seed_pixel[1]
    return (dx**2 + dy**2) ** 0.5


def _geometric_verify_seed(
    seed: SeedAnnotation, matched_pixels: dict[str, tuple[int, int]], residual_threshold: float = 25.0
) -> dict[str, bool]:
    """Fit one affine transform across all of `seed`'s keypoints, flag per-point inliers.

    With fewer than 3 keypoints there's nothing to jointly verify against, so
    every point is trivially marked as an inlier (no other geometry to check it against).
    """
    names = list(matched_pixels.keys())
    if len(names) < 3:
        return {name: True for name in names}

    src_pts = np.array([seed.keypoints[name] for name in names], dtype=float)
    dst_pts = np.array([matched_pixels[name] for name in names], dtype=float)

    fit = fit_affine_ransac(src_pts, dst_pts, residual_threshold=residual_threshold)
    return {name: bool(inlier) for name, inlier in zip(names, fit.inlier_mask)}


def propagate_keypoints(
    model: torch.nn.Module,
    seeds: list[SeedAnnotation],
    target_img: Image.Image,
    confidence_threshold: float = 0.10,
    cycle_error_threshold: float = 40.0,  # pixels, in IMG_SIZE (518) space
    geometric_residual_threshold: float = 25.0,  # pixels, affine-fit residual tolerance
    match_kwargs: dict | None = None,
) -> list[PropagatedKeypoint]:
    """Propagate every keypoint name found across `seeds` onto `target_img`.

    Per seed: match once, transfer every one of its keypoints, then run a
    joint geometric consistency check across that seed's own points. Across
    seeds: for each keypoint name, prefer geometric inliers, then take the
    highest-confidence candidate among those, then run cycle consistency on
    the winner only (expensive, so done once per keypoint name, not per
    candidate).
    """
    match_kwargs = match_kwargs or {}

    # name -> list of (target_pixel, confidence, seed, geometric_inlier)
    candidates: dict[str, list[tuple[tuple[int, int], float, SeedAnnotation, bool]]] = {}

    for seed in seeds:
        result = geco_match(model, seed.image, target_img, **match_kwargs)

        matched_pixels: dict[str, tuple[int, int]] = {}
        confidences: dict[str, float] = {}
        for name, (sx, sy) in seed.keypoints.items():
            match = transfer_keypoint(result, sx, sy, image_size=IMG_SIZE)
            matched_pixels[name] = match.trg_pixel
            confidences[name] = match.confidence

        inlier_flags = _geometric_verify_seed(seed, matched_pixels, geometric_residual_threshold)

        for name in matched_pixels:
            candidates.setdefault(name, []).append(
                (matched_pixels[name], confidences[name], seed, inlier_flags[name])
            )

    propagated: list[PropagatedKeypoint] = []
    for name, votes in candidates.items():
        inlier_votes = [v for v in votes if v[3]]
        pool = inlier_votes if inlier_votes else votes  # fall back to all votes if none are inliers

        best_pixel, best_conf, best_seed, best_inlier = max(pool, key=lambda v: v[1])
        seed_pixel = best_seed.keypoints[name]

        cycle_error = _cycle_consistency_error(
            model, target_img, best_seed.image, best_pixel, seed_pixel, match_kwargs
        )

        accepted = best_conf >= confidence_threshold and cycle_error <= cycle_error_threshold
        propagated.append(
            PropagatedKeypoint(
                name=name,
                pixel=best_pixel,
                confidence=best_conf,
                cycle_error=cycle_error,
                n_votes=len(votes),
                geometric_inlier=best_inlier,
                accepted=accepted,
            )
        )

    return propagated
