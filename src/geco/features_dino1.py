"""DINOv1 model loading and multi-scale feature extraction. Fully ungated
(no HF approval needed) -- a genuinely independent second opinion from an
earlier DINO generation trained on far less data than DINOv2/v3.

Uses forward hooks on `model.blocks[i]` rather than relying on an
undocumented `get_intermediate_layers` method, so extraction doesn't
silently break if this particular hub checkpoint's API differs from
DINOv2's.
"""

from __future__ import annotations

from typing import Literal

import torch
from PIL import Image
from torchvision import transforms

from .features import get_device

# DINOv1 ViT-B/16 uses 16x16 patches; 512 = 32 * 16.
IMG_SIZE_DINO1 = 512
PATCH_SIZE_DINO1 = 16
GRID_SIZE_DINO1 = IMG_SIZE_DINO1 // PATCH_SIZE_DINO1  # 32

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_preprocess = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE_DINO1),
        transforms.CenterCrop(IMG_SIZE_DINO1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def load_dino1(variant: str = "dino_vitb16", device: torch.device | None = None) -> torch.nn.Module:
    """Load a frozen, eval-mode DINOv1 backbone from torch.hub. Fully public,
    no gating, no HF account needed."""
    device = device or get_device()
    model = torch.hub.load("facebookresearch/dino:main", variant)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model, "blocks"):
        raise AttributeError(
            f"Expected model.blocks (ModuleList of transformer blocks) on the loaded "
            f"'{variant}' checkpoint, but it's not present -- the hub repo's ViT "
            "implementation may have changed. Inspect `dir(model)` to find the right "
            "attribute and update extract_multiscale_features_dino1 accordingly."
        )
    return model


def preprocess_image_dino1(img: Image.Image, device: torch.device | None = None) -> torch.Tensor:
    device = device or get_device()
    img = img.convert("RGB")
    return _preprocess(img).unsqueeze(0).to(device)


def extract_multiscale_features_dino1(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    layers: list[int] = (2, 5, 8, 11),
    fusion: Literal["mean", "concat"] = "mean",
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """DINOv1 equivalent of features.extract_multiscale_features, using
    forward hooks on model.blocks[i] instead of get_intermediate_layers.

    DINOv1's ViT has no register tokens (that was introduced later), so only
    1 CLS token needs to be stripped, not 1 + num_register_tokens.
    """
    n_blocks = len(model.blocks)
    for layer_idx in layers:
        if layer_idx >= n_blocks:
            raise ValueError(
                f"Requested layer {layer_idx} but this DINOv1 variant only has "
                f"{n_blocks} blocks (0-indexed, max {n_blocks - 1})."
            )

    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def _make_hook(idx: int):
        def _hook(module, inputs, output):
            captured[idx] = output
        return _hook

    for idx in layers:
        hooks.append(model.blocks[idx].register_forward_hook(_make_hook(idx)))

    try:
        with torch.no_grad():
            model(img_tensor)
    finally:
        for h in hooks:
            h.remove()

    grid = GRID_SIZE_DINO1
    layer_feats: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        feat = captured[layer_idx][:, 1:, :]  # drop CLS token only (no registers in DINOv1)
        b, n, d = feat.shape
        expected = grid * grid
        if n != expected:
            raise ValueError(
                f"Got {n} patch tokens after dropping the CLS token, expected "
                f"{expected} ({grid}x{grid}). Check IMG_SIZE_DINO1/PATCH_SIZE_DINO1 "
                "against this model variant's actual patch size."
            )
        layer_feats[layer_idx] = feat.transpose(-1, -2).reshape(b, d, grid, grid)

    stacked = torch.stack(list(layer_feats.values()), dim=0)
    if fusion == "mean":
        fused = stacked.mean(dim=0)
    elif fusion == "concat":
        fused = torch.cat(list(layer_feats.values()), dim=1)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion!r}")

    return fused, layer_feats
