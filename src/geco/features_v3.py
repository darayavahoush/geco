"""DINOv3 model loading, preprocessing, and multi-scale feature extraction,
mirroring features.py's conventions for DINOv2 but via Hugging Face
`transformers` (no gated torch.hub checkpoint URL needed) and DINOv3's
different patch size (16, not 14).

Requires: pip install transformers --break-system-packages (or without the
flag on Windows), and a Hugging Face account that has clicked "Agree" on
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m (gated but
one-click, not a form-and-wait), plus `huggingface-cli login` or an
HF_TOKEN env var set.

DINOv3 ViT-B/16 has the same embedding dim (768) as the DINOv2 ViT-B/14
`geco_match` already uses, purely coincidentally convenient for later
comparing the two models' outputs side by side (NOT for combining their
raw feature vectors, which would be meaningless across two differently
trained models).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .features import get_device

# DINOv3 ViT-B/16 uses 16x16 patches; 512 = 32 * 16, giving a clean 32x32 grid.
# NOT the same as DINOv2's IMG_SIZE=518/GRID_SIZE=37 in features.py -- these
# are intentionally separate constants, not reused, since the patch size differs.
IMG_SIZE_V3 = 512
PATCH_SIZE_V3 = 16
GRID_SIZE_V3 = IMG_SIZE_V3 // PATCH_SIZE_V3  # 32

DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def load_dinov3(
    model_name: str = DEFAULT_MODEL_NAME, device: torch.device | None = None
) -> tuple[torch.nn.Module, "AutoImageProcessor"]:
    """Load a frozen, eval-mode DINOv3 backbone + its HF image processor.

    Returns (model, processor). The processor is only used to read the
    model's expected normalization stats (image_mean/image_std) -- actual
    resizing is done ourselves via torchvision, matching features.py's
    Resize+CenterCrop pipeline exactly, so map_point_to_model_space's
    coordinate math stays valid for DINOv3 too (just with IMG_SIZE_V3
    instead of IMG_SIZE).
    """
    from transformers import AutoImageProcessor, AutoModel  # local import: optional heavy dep

    device = device or get_device()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def _build_preprocess(processor) -> transforms.Compose:
    mean = processor.image_mean
    std = processor.image_std
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE_V3),
            transforms.CenterCrop(IMG_SIZE_V3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def preprocess_image_v3(
    img: Image.Image, processor, device: torch.device | None = None
) -> torch.Tensor:
    """Convert a PIL image into a normalized [1, 3, 512, 512] tensor, using
    DINOv3's own mean/std but features.py's Resize+CenterCrop convention
    (NOT the HF processor's built-in resizing, so coordinate mapping via
    map_point_to_model_space(..., image_size=IMG_SIZE_V3) stays exact)."""
    device = device or get_device()
    preprocess = _build_preprocess(processor)
    img = img.convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


def extract_multiscale_features_v3(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    layers: list[int] = (2, 5, 8, 11),
    fusion: Literal["mean", "concat"] = "mean",
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """DINOv3 equivalent of features.extract_multiscale_features.

    HF's `hidden_states` tuple is [embeddings, after_block_0, after_block_1,
    ..., after_block_{L-1}], so "layer index i" (matching DINOv2's
    get_intermediate_layers(n=[i]) convention, i.e. output after block i)
    is hidden_states[i + 1] here -- the +1 offset accounts for the leading
    embedding entry that DINOv2's API doesn't expose.

    Patch tokens start after 1 CLS token + `num_register_tokens` register
    tokens (4 for the released ViT variants) -- read from model.config
    rather than hardcoded, in case a variant differs.
    """
    num_register_tokens = getattr(model.config, "num_register_tokens", 4)
    num_prefix_tokens = 1 + num_register_tokens

    with torch.no_grad():
        outputs = model(pixel_values=img_tensor, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    n_available = len(hidden_states) - 1  # number of transformer blocks
    for layer_idx in layers:
        if layer_idx >= n_available:
            raise ValueError(
                f"Requested layer {layer_idx} but this DINOv3 variant only has "
                f"{n_available} blocks (0-indexed, max {n_available - 1}). "
                "Adjust `layers` for this model size."
            )

    grid = GRID_SIZE_V3
    layer_feats: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        feat = hidden_states[layer_idx + 1][:, num_prefix_tokens:, :]  # [1, H*W, D]
        b, n, d = feat.shape
        expected = grid * grid
        if n != expected:
            raise ValueError(
                f"Got {n} patch tokens after stripping {num_prefix_tokens} prefix "
                f"tokens, expected {expected} ({grid}x{grid}). Check "
                "num_register_tokens / IMG_SIZE_V3 / PATCH_SIZE_V3 assumptions "
                "against this model's actual config."
            )
        feat_map = feat.transpose(-1, -2).reshape(b, d, grid, grid)
        layer_feats[layer_idx] = feat_map

    stacked = torch.stack(list(layer_feats.values()), dim=0)
    if fusion == "mean":
        fused = stacked.mean(dim=0)
    elif fusion == "concat":
        fused = torch.cat(list(layer_feats.values()), dim=1)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion!r}")

    return fused, layer_feats
