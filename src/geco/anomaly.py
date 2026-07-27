"""Component 2 / Track B — Correspondence-Guided Anomaly Detection.

Align a test image to one or more known-good "golden reference" images using
Component 1's correspondence engine, then flag regions that are suspect by
combining three signals:

  1. **Dustbin mass** — patches the OT solver couldn't confidently match to
     anything in the reference at all.
  2. **Match cost** — 1 - cosine similarity to the best available reference
     patch, for patches that DID get matched (a bad-but-accepted match still
     raises this).
  3. **Patch appearance cost** — a classical (non-learned) signal: raw pixel
     difference between each test patch and its best-matched reference
     patch. This catches a class of anomaly deep cosine similarity can miss
     (DINOv2 features are fairly texture/color-invariant by design, so two
     patches can look "semantically similar" in feature space while being
     visibly different in raw appearance — e.g. a scratch or discoloration
     on an otherwise structurally normal part). Combining a classical pixel
     signal with the deep semantic one is a deliberate hybrid, not a
     redundant one.

If more than one reference image is supplied, each is aligned separately and
the final anomaly map is the per-patch MEDIAN across references — this makes
the score robust to any single reference being a poor representative of
"normal," rather than trusting one golden image outright.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from .features import PATCH_SIZE, preprocess_image
from .matching import MatchResult, geco_match


@dataclass
class AnomalyResult:
    match: MatchResult  # from the FIRST reference, kept for visualization/debugging
    anomaly_map: torch.Tensor  # [H, W], 0 = matches reference well, 1 = highly anomalous
    dustbin_mask: torch.Tensor  # [H, W] bool, True where majority of references dustbinned this patch
    anomaly_score: float  # scalar summary = mean anomaly_map, useful for pass/fail triage
    n_references: int


def _patch_appearance_cost(
    test_tensor: torch.Tensor, ref_tensor: torch.Tensor, cos_sim: torch.Tensor, grid: int
) -> torch.Tensor:
    """Vectorized per-patch raw-pixel difference between each test patch and its best reference match.

    test_tensor, ref_tensor: [1, 3, 518, 518] normalized image tensors.
    cos_sim: [N, M] from the matching pipeline, used only to find each test
             patch's best-matching reference patch index.
    Returns: [N] cost in ~[0, 1] (normalized-image L1 distance, clamped).
    """
    best_ref_idx = cos_sim.argmax(dim=1)  # [N]

    # Unfold both images into non-overlapping PATCH_SIZE x PATCH_SIZE patches:
    # [1, 3*P*P, num_patches]
    test_patches = F.unfold(test_tensor, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)[0]  # [3*P*P, N]
    ref_patches = F.unfold(ref_tensor, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)[0]  # [3*P*P, M]

    matched_ref_patches = ref_patches[:, best_ref_idx]  # [3*P*P, N], reordered to align with test patches

    diff = (test_patches - matched_ref_patches).abs().mean(dim=0)  # [N]
    cost = (diff / 2.0).clamp(0, 1)  # normalized tensors are roughly in [-2, 2]
    return cost


def _score_against_one_reference(
    model: torch.nn.Module, test_img: Image.Image, reference_img: Image.Image, match_kwargs: dict
) -> tuple[MatchResult, torch.Tensor, torch.Tensor]:
    """Returns (match_result, combined_anomaly_map [H,W], dustbin_fraction [N])."""
    result = geco_match(model, test_img, reference_img, **match_kwargs)
    grid = result.grid_size
    n_patches = grid * grid

    # Row-normalize dustbin mass by each patch's OWN total row mass (same fix as the
    # keypoint-confidence bug: raw transport mass scales as ~1/N and isn't comparable
    # across grid sizes or images). This gives "what fraction of THIS patch's mass went
    # to the dustbin" -- a proper 0-1 value, independent of per-pair rescaling.
    row_sums = result.transport_plan[:n_patches, :].sum(dim=1)
    dustbin_fraction = result.transport_plan[:n_patches, -1] / (row_sums + 1e-8)

    best_sim = result.cos_sim.max(dim=1).values
    match_cost = (1 - best_sim) / 2

    device = next(model.parameters()).device
    test_tensor = preprocess_image(test_img, device)
    ref_tensor = preprocess_image(reference_img, device)
    appearance_cost = _patch_appearance_cost(test_tensor, ref_tensor, result.cos_sim, grid)

    # NOTE: deliberately NOT min-max normalized here. All three components are already
    # naturally scaled to ~[0, 1] (dustbin_fraction and appearance_cost are exact, match_cost
    # is a cosine-derived ratio). Per-pair min-max rescaling would stretch every pair to fill
    # the full 0-1 range regardless of whether it's a great or terrible match overall --
    # destroying the cross-pair comparability that anomaly SCORING requires (a bad pair should
    # read as a higher absolute score than a good pair, not just "highest within itself").
    combined = 0.4 * dustbin_fraction + 0.3 * match_cost + 0.3 * appearance_cost

    return result, combined.reshape(grid, grid), dustbin_fraction


def detect_anomalies(
    model: torch.nn.Module,
    test_img: Image.Image,
    reference_imgs: Image.Image | list[Image.Image],
    match_kwargs: dict | None = None,
) -> AnomalyResult:
    """Score `test_img` for anomalies relative to one or more golden reference images."""
    match_kwargs = match_kwargs or {}
    if isinstance(reference_imgs, Image.Image):
        reference_imgs = [reference_imgs]

    first_result = None
    anomaly_maps = []
    dustbin_masses = []
    for ref in reference_imgs:
        result, anomaly_map, dustbin_mass = _score_against_one_reference(model, test_img, ref, match_kwargs)
        if first_result is None:
            first_result = result
        anomaly_maps.append(anomaly_map)
        dustbin_masses.append(dustbin_mass)

    stacked_maps = torch.stack(anomaly_maps, dim=0)  # [R, H, W]
    median_map = stacked_maps.median(dim=0).values

    stacked_dustbin = torch.stack(dustbin_masses, dim=0)  # [R, N]
    grid = first_result.grid_size
    dustbin_threshold = stacked_dustbin.mean() + 2 * stacked_dustbin.std()
    dustbin_vote = (stacked_dustbin > dustbin_threshold).float().mean(dim=0)  # [N], fraction of refs that flagged it
    dustbin_mask = (dustbin_vote > 0.5).reshape(grid, grid)  # majority vote across references

    return AnomalyResult(
        match=first_result,
        anomaly_map=median_map,
        dustbin_mask=dustbin_mask,
        anomaly_score=median_map.mean().item(),
        n_references=len(reference_imgs),
    )
