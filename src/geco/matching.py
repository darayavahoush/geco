"""Confidence-aware Optimal Transport matching between two feature maps."""

from __future__ import annotations

from dataclasses import dataclass

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
        mass: fraction of total probability mass assigned to real patches
              (the rest goes to the dustbin / no-match slot).

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

    mu = torch.cat([mu_patches, torch.tensor([1 - mass], device=conf_src.device)])
    nu = torch.cat([nu_patches, torch.tensor([1 - mass], device=conf_trg.device)])
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
    matrix -> Sinkhorn OT.

    Args:
        model: a loaded DINOv2 backbone (see features.load_dinov2).
        img_src, img_trg: source/target PIL images.
        layers: which DINOv2 blocks to fuse for multi-scale features.
        alpha: uniform vs. confidence-weighted marginal blend (see get_confidence_aware_marginals).
        z_base, z_range: adaptive dustbin parameters (see compute_adaptive_dustbin).
        reg: Sinkhorn entropic regularization strength (lower = sharper, slower to converge).
    """
    device = next(model.parameters()).device

    src_tensor = preprocess_image(img_src, device)
    trg_tensor = preprocess_image(img_trg, device)

    src_feat, _ = extract_multiscale_features(model, src_tensor, layers=list(layers))
    trg_feat, _ = extract_multiscale_features(model, trg_tensor, layers=list(layers))

    conf_src = compute_patch_confidence_v2(src_feat)
    conf_trg = compute_patch_confidence_v2(trg_feat)

    mu, nu = get_confidence_aware_marginals(conf_src, conf_trg, alpha=alpha)
    dustbin_src = compute_adaptive_dustbin(conf_src, z_base, z_range)
    dustbin_trg = compute_adaptive_dustbin(conf_trg, z_base, z_range)

    b, d, h, w = src_feat.shape
    src_flat = F.normalize(src_feat.reshape(d, -1).T, dim=-1)
    trg_flat = F.normalize(trg_feat.reshape(d, -1).T, dim=-1)
    cos_sim = torch.mm(src_flat, trg_flat.T)  # [N, M]

    # Weight similarity by joint confidence before building the cost matrix.
    conf_matrix = (
        conf_src.flatten().unsqueeze(1) * conf_trg.flatten().unsqueeze(0)
    ).to(cos_sim.device)
    weighted_sim = cos_sim * conf_matrix

    n, m = weighted_sim.shape
    dustbin_col = dustbin_src.unsqueeze(1).expand(n, 1)
    dustbin_row = dustbin_trg.unsqueeze(0).expand(1, m)
    dustbin_corner = torch.tensor([[z_base]], device=weighted_sim.device)

    cost_matrix = torch.cat(
        [
            torch.cat([weighted_sim, dustbin_col], dim=1),
            torch.cat([dustbin_row, dustbin_corner], dim=1),
        ],
        dim=0,
    )  # [N+1, M+1]

    cost_np = (2 - cost_matrix).cpu().float().numpy()
    transport_plan = ot.sinkhorn(
        mu.cpu().float().numpy(),
        nu.cpu().float().numpy(),
        cost_np,
        reg=reg,
        numItermax=500,
        stopThr=1e-9,
    )
    transport_plan = torch.tensor(transport_plan, device=device)

    entropy = -(transport_plan * torch.log(transport_plan + 1e-8)).sum().item()
    mean_conf = transport_plan[:-1, :-1].max(dim=1)[0].mean().item()

    return MatchResult(
        transport_plan=transport_plan,
        cos_sim=cos_sim,
        conf_src=conf_src,
        conf_trg=conf_trg,
        grid_size=h,
        entropy=entropy,
        mean_confidence=mean_conf,
    )
