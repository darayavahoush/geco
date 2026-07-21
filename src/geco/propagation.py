"""Component 2 / Track A — Sparse-to-Dense Annotation Propagation.

Given a small set of manually keypoint-annotated "seed" images, propagate
those keypoint labels onto an unlabeled target image using Component 1's
correspondence engine, without touching its internals.

Two things beyond a naive single-seed lookup:

1. **Multi-seed voting** — several seed images may match the target with
   different confidence; we take the confidence-weighted consensus rather
   than trusting a single reference.
2. **Cycle consistency** — for each candidate propagated point, we match
   target -> seed and check it lands back near the original seed keypoint.
   Points that fail this check are flagged as unreliable independent of the
   raw OT confidence score, which catches a class of errors (near-symmetric
   parts, repeated textures) that confidence alone misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

from .features import IMG_SIZE
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
    n_votes: int  # how many seeds contributed
    accepted: bool  # True if it passed the confidence + cycle-consistency thresholds


def _cycle_consistency_error(
    model: torch.nn.Module,
    target_img: Image.Image,
    seed_img: Image.Image,
    target_pixel: tuple[int, int],
    seed_pixel: tuple[int, int],
    match_kwargs: dict,
) -> float:
    """Match target -> seed and measure how far the round trip lands from the original seed point."""
    back_result = geco_match(model, target_img, seed_img, **match_kwargs)
    back_match = transfer_keypoint(back_result, target_pixel[0], target_pixel[1], image_size=IMG_SIZE)
    dx = back_match.trg_pixel[0] - seed_pixel[0]
    dy = back_match.trg_pixel[1] - seed_pixel[1]
    return (dx**2 + dy**2) ** 0.5


def propagate_keypoints(
    model: torch.nn.Module,
    seeds: list[SeedAnnotation],
    target_img: Image.Image,
    confidence_threshold: float = 0.15,
    cycle_error_threshold: float = 40.0,  # pixels, in IMG_SIZE (518) space
    match_kwargs: dict | None = None,
) -> list[PropagatedKeypoint]:
    """Propagate every keypoint name found across `seeds` onto `target_img`.

    For each keypoint name, matches every seed that has it, keeps the
    highest-confidence candidate as the vote, runs a cycle-consistency check
    on that candidate, and marks it accepted only if it clears both
    thresholds. Returns one PropagatedKeypoint per keypoint name.
    """
    match_kwargs = match_kwargs or {}

    # keypoint name -> list of (result, seed, pixel_in_target, confidence)
    candidates: dict[str, list[tuple[MatchResult, SeedAnnotation, tuple[int, int], float]]] = {}

    for seed in seeds:
        result = geco_match(model, seed.image, target_img, **match_kwargs)
        for name, (sx, sy) in seed.keypoints.items():
            match = transfer_keypoint(result, sx, sy, image_size=IMG_SIZE)
            candidates.setdefault(name, []).append(
                (result, seed, match.trg_pixel, match.confidence)
            )

    propagated: list[PropagatedKeypoint] = []
    for name, votes in candidates.items():
        best_result, best_seed, best_pixel, best_conf = max(votes, key=lambda v: v[3])
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
                accepted=accepted,
            )
        )

    return propagated
