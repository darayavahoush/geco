"""Component 2 / Track B — Correspondence-Guided Anomaly Detection.

Align a test image to a known-good "golden reference" image using Component
1's correspondence engine, then flag regions that either:

  (a) got routed to the OT dustbin (no confident match found at all), or
  (b) matched, but only at high transport cost (best available match is a
      poor one) — i.e. present in the reference but locally *different* in
      the test image.

This gives a per-patch anomaly score without ever training a defect
classifier: anything that doesn't correspond well to the golden reference is
suspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from .matching import MatchResult, geco_match


@dataclass
class AnomalyResult:
    match: MatchResult
    anomaly_map: torch.Tensor  # [H, W], 0 = matches reference well, 1 = highly anomalous
    dustbin_mask: torch.Tensor  # [H, W] bool, True where patch was routed to the dustbin
    anomaly_score: float  # scalar summary = mean anomaly_map, useful for pass/fail triage


def detect_anomalies(
    model: torch.nn.Module,
    test_img: Image.Image,
    reference_img: Image.Image,
    match_kwargs: dict | None = None,
) -> AnomalyResult:
    """Compute a per-patch anomaly map for `test_img` relative to `reference_img`.

    anomaly_map is derived from two signals, combined and normalized to [0, 1]:
      1. dustbin mass — how much of a test patch's transport mass went to "no match"
      2. match cost — 1 - (best-match cosine similarity to the reference), for
         patches that DID find a match, so a bad-but-accepted match still
         raises the score.
    """
    match_kwargs = match_kwargs or {}
    # test image is the "source" (the one we're inspecting); reference is the "target".
    result = geco_match(model, test_img, reference_img, **match_kwargs)

    grid = result.grid_size
    n_patches = grid * grid

    # Dustbin mass per test patch: last column of the transport plan (excluding dustbin row).
    dustbin_mass = result.transport_plan[:n_patches, -1]  # [N]
    dustbin_mass = dustbin_mass / (dustbin_mass.max() + 1e-8)

    # Best-match cost per test patch, over real (non-dustbin) reference patches only.
    best_sim = result.cos_sim.max(dim=1).values  # [N], in roughly [-1, 1]
    match_cost = (1 - best_sim) / 2  # normalize to ~[0, 1]

    combined = 0.5 * dustbin_mass + 0.5 * match_cost
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    anomaly_map = combined.reshape(grid, grid)

    dustbin_threshold = dustbin_mass.mean() + 2 * dustbin_mass.std()
    dustbin_mask = (dustbin_mass > dustbin_threshold).reshape(grid, grid)

    return AnomalyResult(
        match=result,
        anomaly_map=anomaly_map,
        dustbin_mask=dustbin_mask,
        anomaly_score=anomaly_map.mean().item(),
    )
