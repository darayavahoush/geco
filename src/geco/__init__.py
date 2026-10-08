"""
GECO-Enhanced: Confidence-Aware Semantic Correspondence
=========================================================

A DINOv2-based pipeline for finding semantic keypoint correspondences between
two images (e.g. "where is this bird's eye in a different photo of a bird").

Core idea over vanilla GECO:
  1. Extract multi-scale DINOv2 features (fuse several transformer layers
     instead of using only the last one).
  2. Estimate a per-patch *confidence* score (how distinctive vs. generic a
     patch's feature is).
  3. Feed that confidence into the Optimal Transport (Sinkhorn) matching as
     both the marginal distributions AND a per-patch adaptive "dustbin"
     threshold, so ambiguous/background patches are easier to discard and
     distinctive patches are matched more aggressively.

See README.md for the full pipeline diagram and usage examples.
"""

try:
    from .features import load_dinov2, preprocess_image, extract_multiscale_features
    from .confidence import (
        compute_patch_confidence,
        compute_patch_confidence_v2,
        compute_adaptive_dustbin,
    )
    from .matching import get_confidence_aware_marginals, geco_match
    from .keypoints import transfer_keypoint
    from .geometry import fit_affine_ransac, AffineFitResult
    from .propagation import SeedAnnotation, PropagatedKeypoint, propagate_keypoints
    from .anomaly import AnomalyResult, detect_anomalies
except (ImportError, Exception):
    pass

__all__ = [
    "load_dinov2",
    "preprocess_image",
    "extract_multiscale_features",
    "compute_patch_confidence",
    "compute_patch_confidence_v2",
    "compute_adaptive_dustbin",
    "get_confidence_aware_marginals",
    "geco_match",
    "transfer_keypoint",
    "fit_affine_ransac",
    "AffineFitResult",
    "SeedAnnotation",
    "PropagatedKeypoint",
    "propagate_keypoints",
    "AnomalyResult",
    "detect_anomalies",
]

__version__ = "0.1.0"
