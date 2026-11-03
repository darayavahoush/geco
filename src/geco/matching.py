"""Confidence-aware Optimal Transport matching between two feature maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
import torch
import torch.nn.functional as F
from PIL import Image

from .confidence import compute_adaptive_dustbin, compute_patch_confidence_v2
from .features import extract_multiscale_features, preprocess_image


def get_confidence_aware_marginals(
    conf_src: torch.Tensor, conf_trg: torch.Tensor, alpha: float = 0.5, mass: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend uniform and confidence-weighted OT marginals, with a dustbin slot.

    Args:
        conf_src, conf_trg: [H, W] confidence maps.
        alpha: 0 = fully uniform marginals (vanilla OT), 1 = fully confidence-weighted.
        mass: kept for API compatibility. It has no effect: the dustbin mass is
              defined relative to the patch mass and both vectors are renormalised.

    Returns:
        mu: [N+1] source marginal (N patches + 1 dustbin).
        nu: [M+1] target marginal (M patches + 1 dustbin).
    """
    n, m = conf_src.numel(), conf_trg.numel()
    conf_src_flat, conf_trg_flat = conf_src.flatten(), conf_trg.flatten()

    uniform_src = torch.ones(n, device=conf_src.device) / n
    uniform_trg = torch.ones(m, device=conf_trg.device) / m

    conf_src_norm = conf_src_flat / (conf_src_flat.sum() + 1e-8)
    conf_trg_norm = conf_trg_flat / (conf_trg_flat.sum() + 1e-8)

    mu_patches = ((1 - alpha) * uniform_src + alpha * conf_src_norm) * mass
    nu_patches = ((1 - alpha) * uniform_trg + alpha * conf_trg_norm) * mass

    # The dustbin's marginal mass must be comparable to ONE patch's mass, not a fixed
    # lump of the whole distribution — otherwise it dominates every row/column of the
    # transport plan regardless of match quality, and everything gets routed there.
    src_dustbin_mass = mu_patches.mean()
    trg_dustbin_mass = nu_patches.mean()

    mu = torch.cat([mu_patches, src_dustbin_mass.reshape(1)])
    nu = torch.cat([nu_patches, trg_dustbin_mass.reshape(1)])
    mu = mu / mu.sum()
    nu = nu / nu.sum()
    return mu, nu


@dataclass
class MatchResult:
    transport_plan: torch.Tensor  # [N+1, M+1], last row/col = dustbin
    cos_sim: torch.Tensor  # [N, M] raw cosine similarity
    conf_src: torch.Tensor  # [H, W]
    conf_trg: torch.Tensor  # [H, W]
    grid_size: int
    entropy: float
    mean_confidence: float


def _solve_sinkhorn(mu: torch.Tensor, nu: torch.Tensor, cost: torch.Tensor, reg: float) -> torch.Tensor:
    """Entropic OT plan, robust to underflow.

    The plan is invariant to adding a constant to the cost (marginals are fixed),
    so we shift the cost to start at 0. Without this, exp(-C/reg) with C~1-2 and
    reg=0.05 is ~1e-18 and the scaling vectors can overflow in float32. We solve
    in float64 and fall back to the log-domain solver if the result is not a
    finite, marginal-consistent plan.
    """
    mu_np = mu.detach().cpu().double().numpy()
    nu_np = nu.detach().cpu().double().numpy()
    cost_np = cost.detach().cpu().double().numpy()
    cost_np = cost_np - cost_np.min()

    plan = ot.sinkhorn(mu_np, nu_np, cost_np, reg=reg, numItermax=1000, stopThr=1e-9)
    ok = np.isfinite(plan).all() and abs(plan.sum() - 1.0) < 1e-3
    if ok:
        ok = np.abs(plan.sum(axis=1) - mu_np).max() < 1e-3 * max(1.0, mu_np.max() * 1e3)
    if not ok:
        plan = ot.sinkhorn(mu_np, nu_np, cost_np, reg=reg, numItermax=2000, stopThr=1e-9, method="sinkhorn_log")
    return torch.from_numpy(np.asarray(plan, dtype=np.float32))


def match_from_features(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    alpha: float = 0.8,
    z_base: float = 0.3,
    z_range: float = 0.25,
    reg: float = 0.05,
) -> MatchResult:
    """Confidence-aware OT matching from already-extracted [1, D, H, W] feature
    maps. Shared by every backbone (DINOv1/v2/v3, CLIP) so numerical fixes apply
    everywhere.
    """
    device = src_feat.device
    conf_src = compute_patch_confidence_v2(src_feat)
    conf_trg = compute_patch_confidence_v2(trg_feat)

    mu, nu = get_confidence_aware_marginals(conf_src, conf_trg, alpha=alpha)
    dustbin_src = compute_adaptive_dustbin(conf_src, z_base, z_range)
    dustbin_trg = compute_adaptive_dustbin(conf_trg, z_base, z_range)

    _, d, h, _ = src_feat.shape
    src_flat = F.normalize(src_feat.reshape(d, -1).T, dim=-1)
    trg_flat = F.normalize(trg_feat.reshape(d, -1).T, dim=-1)
    cos_sim = torch.mm(src_flat, trg_flat.T)  # [N, M]

    # Weight similarity by joint confidence before building the cost matrix.
    conf_matrix = (conf_src.flatten().unsqueeze(1) * conf_trg.flatten().unsqueeze(0)).to(cos_sim.device)
    weighted_sim = cos_sim * conf_matrix

    n, m = weighted_sim.shape
    dustbin_col = dustbin_src.unsqueeze(1).expand(n, 1)
    dustbin_row = dustbin_trg.unsqueeze(0).expand(1, m)
    dustbin_corner = torch.tensor([[z_base]], device=weighted_sim.device, dtype=weighted_sim.dtype)

    score = torch.cat(
        [
            torch.cat([weighted_sim, dustbin_col], dim=1),
            torch.cat([dustbin_row, dustbin_corner], dim=1),
        ],
        dim=0,
    )  # [N+1, M+1]

    transport_plan = _solve_sinkhorn(mu, nu, 2 - score, reg).to(device)

    entropy = -(transport_plan * torch.log(transport_plan + 1e-8)).sum().item()
    patch_rows = transport_plan[:-1, :-1]  # exclude the dustbin row/column
    row_sums = patch_rows.sum(dim=1, keepdim=True)
    mean_conf = (patch_rows / (row_sums + 1e-8)).max(dim=1)[0].mean().item()

    return MatchResult(
        transport_plan=transport_plan,
        cos_sim=cos_sim,
        conf_src=conf_src,
        conf_trg=conf_trg,
        grid_size=h,
        entropy=entropy,
        mean_confidence=mean_conf,
    )


def geco_match(
    model: torch.nn.Module,
    img_src: Image.Image,
    img_trg: Image.Image,
    layers: list[int] = (2, 5, 8, 11),
    alpha: float = 0.8,
    z_base: float = 0.3,
    z_range: float = 0.25,
    reg: float = 0.05,
) -> MatchResult:
    """Full confidence-aware GECO matching pipeline between two images.

    Steps: multi-scale feature extraction -> per-patch confidence -> confidence
    weighted OT marginals + adaptive dustbin -> confidence-weighted cosine cost
    matrix -> Sinkhorn OT (see match_from_features).

    Args:
        model: a loaded DINOv2 backbone (see features.load_dinov2).
        img_src, img_trg: source/target PIL images.
        layers: DINOv2 blocks to fuse for multi-scale features.
        alpha, z_base, z_range, reg: see match_from_features.
    """
    device = next(model.parameters()).device
    src_feat, _ = extract_multiscale_features(model, preprocess_image(img_src, device), layers=list(layers))
    trg_feat, _ = extract_multiscale_features(model, preprocess_image(img_trg, device), layers=list(layers))
    return match_from_features(src_feat, trg_feat, alpha, z_base, z_range, reg)
