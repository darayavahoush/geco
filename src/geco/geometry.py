"""Lightweight RANSAC affine fitting, used by Track A for geometric verification.

Implemented from scratch with numpy only (no opencv/skimage dependency) since
this is a self-contained part of the pipeline's own logic, not an off-the-shelf
call — worth having transparent, inspectable code for a course project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AffineFitResult:
    matrix: np.ndarray | None  # [2, 3] affine matrix, or None if fit failed
    inlier_mask: np.ndarray  # [N] bool
    residuals: np.ndarray  # [N] float, pixel distance between predicted and actual point


def _fit_affine_exact(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares affine fit for >=3 point pairs. src, dst: [N, 2]."""
    n = src.shape[0]
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = src[i]
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = dst[i, 0]
        b[2 * i + 1] = dst[i, 1]
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    return params.reshape(2, 3)  # rows: [a,b,tx], [c,d,ty]


def _apply_affine(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1))
    homog = np.hstack([pts, ones])  # [N, 3]
    return homog @ matrix.T  # [N, 2]


def fit_affine_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    residual_threshold: float = 25.0,
    n_iters: int = 300,
    seed: int = 0,
) -> AffineFitResult:
    """Fit src_pts -> dst_pts with an affine transform, robust to outlier correspondences.

    Needs at least 3 point pairs. With fewer than 4 pairs, RANSAC subsampling is
    skipped (nothing to resample) and this just returns the exact-fit residuals.
    """
    n = src_pts.shape[0]
    if n < 3:
        return AffineFitResult(matrix=None, inlier_mask=np.ones(n, dtype=bool), residuals=np.zeros(n))

    rng = np.random.default_rng(seed)

    if n == 3:
        matrix = _fit_affine_exact(src_pts, dst_pts)
        residuals = np.linalg.norm(_apply_affine(matrix, src_pts) - dst_pts, axis=1)
        return AffineFitResult(matrix=matrix, inlier_mask=residuals < residual_threshold, residuals=residuals)

    best_inliers = None
    best_count = -1
    for _ in range(n_iters):
        sample_idx = rng.choice(n, size=3, replace=False)
        try:
            candidate = _fit_affine_exact(src_pts[sample_idx], dst_pts[sample_idx])
        except np.linalg.LinAlgError:
            continue
        residuals = np.linalg.norm(_apply_affine(candidate, src_pts) - dst_pts, axis=1)
        inliers = residuals < residual_threshold
        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 3:
        # Couldn't find a consistent model at all -> everything is geometrically suspect.
        matrix = _fit_affine_exact(src_pts, dst_pts)
        residuals = np.linalg.norm(_apply_affine(matrix, src_pts) - dst_pts, axis=1)
        return AffineFitResult(matrix=matrix, inlier_mask=np.zeros(n, dtype=bool), residuals=residuals)

    # Refit using all inliers from the best RANSAC round for a cleaner final model.
    matrix = _fit_affine_exact(src_pts[best_inliers], dst_pts[best_inliers])
    residuals = np.linalg.norm(_apply_affine(matrix, src_pts) - dst_pts, axis=1)
    return AffineFitResult(matrix=matrix, inlier_mask=residuals < residual_threshold, residuals=residuals)
