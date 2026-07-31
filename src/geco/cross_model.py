"""Cross-model correspondence: run the SAME confidence-aware OT matching
logic on two architecturally-different frozen backbones (DINOv2 and
DINOv3) and compare their independent predictions.

Core idea under test: does AGREEMENT between two independently-trained
models predict correctness better than either model's own self-reported
confidence? A single model's confidence is self-assessed and can be
confidently wrong (fooled by symmetry, degraded features on texture-poor
surfaces, etc. -- see the bottle/chair zero-shot collapse). Two models
that disagree are flagging real uncertainty neither one would catch alone.

This module deliberately does NOT combine or fuse DINOv2's and DINOv3's
raw feature vectors (that would be the "SD+DINO"-style fusion idea, a
different approach) -- it runs two completely independent OT solves and
only compares their final PIXEL predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import ot
import torch
import torch.nn.functional as F
from PIL import Image

from .confidence import compute_adaptive_dustbin, compute_patch_confidence_v2
from .features import IMG_SIZE, get_device, map_point_to_model_space
from .features import preprocess_image as preprocess_image_v2
from .features import extract_multiscale_features as extract_multiscale_features_v2
from .features_v3 import (
    GRID_SIZE_V3,
    IMG_SIZE_V3,
    extract_multiscale_features_v3,
    preprocess_image_v3,
)
from .matching import MatchResult, get_confidence_aware_marginals


def map_point_from_model_space(
    mx: float, my: float, orig_width: int, orig_height: int, image_size: int
) -> tuple[float, float]:
    """Inverse of features.map_point_to_model_space: model-space pixel ->
    original image pixel. Needed to compare DINOv2's and DINOv3's
    predictions (made in two DIFFERENT model spaces, 518 vs 512) in a
    common coordinate system.
    """
    scale = image_size / min(orig_width, orig_height)
    new_w, new_h = orig_width * scale, orig_height * scale
    crop_x0 = (new_w - image_size) / 2
    crop_y0 = (new_h - image_size) / 2
    orig_x = (mx + crop_x0) / scale
    orig_y = (my + crop_y0) / scale
    return orig_x, orig_y


def _match_from_features(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    grid_size: int,
    alpha: float,
    z_base: float,
    z_range: float,
    reg: float,
) -> MatchResult:
    """The shared tail of geco_match (confidence -> marginals -> cost matrix ->
    Sinkhorn), factored out so both DINOv2 and DINOv3 feature maps can feed
    the exact same OT logic. Mirrors matching.geco_match line for line from
    the point features are already extracted.
    """
    device = src_feat.device
    conf_src = compute_patch_confidence_v2(src_feat)
    conf_trg = compute_patch_confidence_v2(trg_feat)

    mu, nu = get_confidence_aware_marginals(conf_src, conf_trg, alpha=alpha)
    dustbin_src = compute_adaptive_dustbin(conf_src, z_base, z_range)
    dustbin_trg = compute_adaptive_dustbin(conf_trg, z_base, z_range)

    b, d, h, w = src_feat.shape
    src_flat = F.normalize(src_feat.reshape(d, -1).T, dim=-1)
    trg_flat = F.normalize(trg_feat.reshape(d, -1).T, dim=-1)
    cos_sim = torch.mm(src_flat, trg_flat.T)

    conf_matrix = (conf_src.flatten().unsqueeze(1) * conf_trg.flatten().unsqueeze(0)).to(
        cos_sim.device
    )
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
    )

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
    patch_rows = transport_plan[:-1, :-1]
    row_sums = patch_rows.sum(dim=1, keepdim=True)
    normalized_rows = patch_rows / (row_sums + 1e-8)
    mean_conf = normalized_rows.max(dim=1)[0].mean().item()

    return MatchResult(
        transport_plan=transport_plan,
        cos_sim=cos_sim,
        conf_src=conf_src,
        conf_trg=conf_trg,
        grid_size=grid_size,
        entropy=entropy,
        mean_confidence=mean_conf,
    )


def geco_match_v2(
    model: torch.nn.Module,
    img_src: Image.Image,
    img_trg: Image.Image,
    layers: list[int] = (2, 5, 8, 11),
    alpha: float = 0.8,
    z_base: float = 0.3,
    z_range: float = 0.25,
    reg: float = 0.05,
) -> MatchResult:
    """Identical to matching.geco_match -- re-exposed here so the cross-model
    eval script can call a "_v2" / "_v3" pair with parallel signatures."""
    device = next(model.parameters()).device
    src_tensor = preprocess_image_v2(img_src, device)
    trg_tensor = preprocess_image_v2(img_trg, device)
    src_feat, _ = extract_multiscale_features_v2(model, src_tensor, layers=list(layers))
    trg_feat, _ = extract_multiscale_features_v2(model, trg_tensor, layers=list(layers))
    b, d, h, w = src_feat.shape
    return _match_from_features(src_feat, trg_feat, h, alpha, z_base, z_range, reg)


def geco_match_v3(
    model: torch.nn.Module,
    processor,
    img_src: Image.Image,
    img_trg: Image.Image,
    layers: list[int] = (2, 5, 8, 11),
    alpha: float = 0.8,
    z_base: float = 0.3,
    z_range: float = 0.25,
    reg: float = 0.05,
) -> MatchResult:
    """DINOv3 equivalent of geco_match, using features_v3's extraction."""
    device = next(model.parameters()).device
    src_tensor = preprocess_image_v3(img_src, processor, device)
    trg_tensor = preprocess_image_v3(img_trg, processor, device)
    src_feat, _ = extract_multiscale_features_v3(model, src_tensor, layers=list(layers))
    trg_feat, _ = extract_multiscale_features_v3(model, trg_tensor, layers=list(layers))
    b, d, h, w = src_feat.shape
    return _match_from_features(src_feat, trg_feat, h, alpha, z_base, z_range, reg)


def geco_match_dino1(
    model: torch.nn.Module,
    img_src: Image.Image,
    img_trg: Image.Image,
    layers: list[int] = (2, 5, 8, 11),
    alpha: float = 0.8,
    z_base: float = 0.3,
    z_range: float = 0.25,
    reg: float = 0.05,
) -> MatchResult:
    """DINOv1 equivalent of geco_match. Fully ungated backbone."""
    from .features_dino1 import extract_multiscale_features_dino1, preprocess_image_dino1

    device = next(model.parameters()).device
    src_tensor = preprocess_image_dino1(img_src, device)
    trg_tensor = preprocess_image_dino1(img_trg, device)
    src_feat, _ = extract_multiscale_features_dino1(model, src_tensor, layers=list(layers))
    trg_feat, _ = extract_multiscale_features_dino1(model, trg_tensor, layers=list(layers))
    b, d, h, w = src_feat.shape
    return _match_from_features(src_feat, trg_feat, h, alpha, z_base, z_range, reg)


def geco_match_clip(
    model: torch.nn.Module,
    processor,
    img_src: Image.Image,
    img_trg: Image.Image,
    layers: list[int] = (2, 5, 8, 11),
    alpha: float = 0.8,
    z_base: float = 0.3,
    z_range: float = 0.25,
    reg: float = 0.05,
) -> MatchResult:
    """CLIP equivalent of geco_match. Fully ungated backbone, different
    training objective entirely (contrastive image-text, not self-distillation)."""
    from .features_clip import extract_multiscale_features_clip, preprocess_image_clip

    device = next(model.parameters()).device
    src_tensor = preprocess_image_clip(img_src, processor, device)
    trg_tensor = preprocess_image_clip(img_trg, processor, device)
    src_feat, _ = extract_multiscale_features_clip(model, src_tensor, layers=list(layers))
    trg_feat, _ = extract_multiscale_features_clip(model, trg_tensor, layers=list(layers))
    b, d, h, w = src_feat.shape
    return _match_from_features(src_feat, trg_feat, h, alpha, z_base, z_range, reg)


def transfer_and_map(
    result: MatchResult,
    image_size: int,
    src_pixel_orig: tuple[float, float],
    src_width: int,
    src_height: int,
    trg_width: int,
    trg_height: int,
) -> tuple[tuple[float, float], float, bool] | None:
    """Generic version of the transfer used inside compare_models_on_keypoint,
    usable for ANY number/combination of models (v2, dino1, clip, v3, ...).

    Returns (trg_pixel_in_ORIGINAL_image_space, confidence, is_dustbin), or
    None if the source keypoint fell outside this model's crop.
    """
    from .keypoints import transfer_keypoint

    src_model_space = map_point_to_model_space(
        *src_pixel_orig, src_width, src_height, image_size=image_size
    )
    if src_model_space is None:
        return None

    match = transfer_keypoint(result, *src_model_space, image_size=image_size)
    trg_orig = map_point_from_model_space(*match.trg_pixel, trg_width, trg_height, image_size)
    return trg_orig, match.confidence, match.is_dustbin


def mean_pairwise_distance(points: list[tuple[float, float]]) -> float:
    """Mean pixel distance across every pair of predicted target points from
    N independent models -- the N-model generalization of "agreement".
    Lower = more consensus among models.
    """
    if len(points) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            total += (dx**2 + dy**2) ** 0.5
            count += 1
    return total / count


@dataclass
class CrossModelPrediction:
    v2_trg_pixel: tuple[float, float]  # in ORIGINAL image pixel space
    v3_trg_pixel: tuple[float, float]  # in ORIGINAL image pixel space
    v2_confidence: float
    v3_confidence: float
    v2_is_dustbin: bool
    v3_is_dustbin: bool
    agreement_px: float  # pixel distance between the two models' predictions, original-image space


def compare_models_on_keypoint(
    result_v2: MatchResult,
    result_v3: MatchResult,
    src_pixel_orig: tuple[float, float],
    src_width: int,
    src_height: int,
    trg_width: int,
    trg_height: int,
) -> CrossModelPrediction:
    """Transfer one source keypoint (given in ORIGINAL src image pixel space)
    through both models independently, map both predictions back to
    ORIGINAL target image pixel space, and measure their agreement.
    """
    from .keypoints import transfer_keypoint  # local import: avoid import cycle risk

    src_v2 = map_point_to_model_space(*src_pixel_orig, src_width, src_height, image_size=IMG_SIZE)
    src_v3 = map_point_to_model_space(
        *src_pixel_orig, src_width, src_height, image_size=IMG_SIZE_V3
    )
    if src_v2 is None or src_v3 is None:
        raise ValueError("Source keypoint fell outside the model crop for v2 or v3.")

    match_v2 = transfer_keypoint(result_v2, *src_v2, image_size=IMG_SIZE)
    match_v3 = transfer_keypoint(result_v3, *src_v3, image_size=IMG_SIZE_V3)

    v2_orig = map_point_from_model_space(*match_v2.trg_pixel, trg_width, trg_height, IMG_SIZE)
    v3_orig = map_point_from_model_space(*match_v3.trg_pixel, trg_width, trg_height, IMG_SIZE_V3)

    agreement_px = ((v2_orig[0] - v3_orig[0]) ** 2 + (v2_orig[1] - v3_orig[1]) ** 2) ** 0.5

    return CrossModelPrediction(
        v2_trg_pixel=v2_orig,
        v3_trg_pixel=v3_orig,
        v2_confidence=match_v2.confidence,
        v3_confidence=match_v3.confidence,
        v2_is_dustbin=match_v2.is_dustbin,
        v3_is_dustbin=match_v3.is_dustbin,
        agreement_px=agreement_px,
    )
