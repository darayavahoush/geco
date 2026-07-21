"""Rendering helpers that turn tensors into base64-encoded PNGs for the API layer."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def pca_feature_map_to_rgb(feature_map: torch.Tensor) -> np.ndarray:
    """Project a [1, D, H, W] feature map to 3 principal components -> an RGB image array.

    Lets you *see* what the DINOv2 features look like spatially: patches with
    similar features get similar colors.
    """
    b, d, h, w = feature_map.shape
    feat = feature_map.reshape(d, h * w).T.cpu().numpy()  # [H*W, D]

    pca = PCA(n_components=3)
    proj = pca.fit_transform(feat)  # [H*W, 3]
    proj = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + 1e-8)
    return proj.reshape(h, w, 3)


def render_confidence_heatmap(confidence: torch.Tensor, title: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(confidence.cpu().numpy(), cmap="hot", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _fig_to_base64(fig)


def render_pca_map(feature_map: torch.Tensor, title: str) -> str:
    rgb = pca_feature_map_to_rgb(feature_map)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    return _fig_to_base64(fig)


def render_match_heatmap(match_row: torch.Tensor, grid_size: int, title: str) -> str:
    """Visualize where a single source patch's transport mass lands in the target grid."""
    heat = match_row[:-1].reshape(grid_size, grid_size).cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heat, cmap="hot")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _fig_to_base64(fig)
