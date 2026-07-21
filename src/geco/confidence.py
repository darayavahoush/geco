"""Per-patch confidence estimation.

Two variants are implemented (kept from the original notebook exploration):

- v1 (entropy-based): a patch is "confident" if it's similar to only a few
  other patches (peaked similarity distribution = low entropy = distinctive).
  Background patches tend to look like many other background patches, so
  they get high entropy = low confidence.

- v2 (distance-from-mean + smoothing): a patch is "confident" if its feature
  is far from the image's global mean feature (i.e. it stands out), then the
  result is spatially smoothed with a 3x3 average pool to reduce noise.
  This is the version used by the default matching pipeline (`geco_match`)
  since it was empirically more stable across the bird test images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_patch_confidence(feature_map: torch.Tensor) -> torch.Tensor:
    """Entropy-based patch confidence.

    Args:
        feature_map: [1, D, H, W]

    Returns:
        confidence: [H, W], values in [0, 1], higher = more confident/distinctive.
    """
    b, d, h, w = feature_map.shape
    feat = feature_map.reshape(b, d, h * w).transpose(1, 2)  # [1, H*W, D]
    feat = F.normalize(feat, dim=-1)

    sim = torch.bmm(feat, feat.transpose(1, 2))[0]  # [H*W, H*W]
    prob = F.softmax(sim, dim=-1)

    entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1)  # [H*W]
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    confidence = 1.0 - entropy
    return confidence.reshape(h, w)


def compute_patch_confidence_v2(feature_map: torch.Tensor) -> torch.Tensor:
    """Distance-from-global-mean confidence, spatially smoothed.

    Args:
        feature_map: [1, D, H, W]

    Returns:
        confidence: [H, W], values in [0, 1], higher = more confident/distinctive.
    """
    b, d, h, w = feature_map.shape
    feat = feature_map.reshape(b, d, h * w).transpose(1, 2)  # [1, H*W, D]
    feat = F.normalize(feat, dim=-1)

    global_mean = feat.mean(dim=1, keepdim=True)
    diff = (feat - global_mean).norm(dim=-1)[0]  # [H*W]

    confidence = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    confidence = F.avg_pool2d(
        confidence.reshape(1, 1, h, w), kernel_size=3, stride=1, padding=1
    ).reshape(h, w)

    return confidence


def compute_adaptive_dustbin(
    confidence_map: torch.Tensor, z_base: float = 0.3, z_range: float = 0.15
) -> torch.Tensor:
    """Per-patch adaptive "dustbin" (no-match) score.

    Standard GECO uses one fixed dustbin score for every patch. Here, low
    confidence patches get a *lower* dustbin score, making it easier for the
    Sinkhorn solver to route them to the dustbin (i.e. "no confident match")
    instead of forcing a bad correspondence.

    Args:
        confidence_map: [H, W], values in [0, 1].
        z_base: dustbin score for a maximally confident patch.
        z_range: how much the score is lowered for a zero-confidence patch.

    Returns:
        dustbin_scores: [H*W]
    """
    conf_flat = confidence_map.flatten()
    return z_base - z_range * (1 - conf_flat)
