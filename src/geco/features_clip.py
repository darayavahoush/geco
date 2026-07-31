"""CLIP vision-tower loading and multi-scale feature extraction. Fully
ungated. Trained with contrastive image-text supervision -- a completely
different objective than any DINO generation's self-distillation, making
it the strongest "genuinely independent" model for the cross-model
agreement experiment.
"""

from __future__ import annotations

from typing import Literal

import torch
from PIL import Image
from torchvision import transforms

from .features import get_device

# CLIP ViT-B/16 uses 16x16 patches at native 224x224 resolution (position
# embeddings aren't trivially extendable without interpolation, so unlike
# the DINO family we keep this at CLIP's native training resolution).
IMG_SIZE_CLIP = 224
PATCH_SIZE_CLIP = 16
GRID_SIZE_CLIP = IMG_SIZE_CLIP // PATCH_SIZE_CLIP  # 14

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch16"


def load_clip(
    model_name: str = DEFAULT_CLIP_MODEL, device: torch.device | None = None
):
    """Load a frozen, eval-mode CLIP vision tower + its HF image processor.
    Ungated -- no HF access request needed."""
    from transformers import CLIPImageProcessor, CLIPVisionModel

    device = device or get_device()
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def _build_preprocess(processor) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE_CLIP),
            transforms.CenterCrop(IMG_SIZE_CLIP),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


def preprocess_image_clip(
    img: Image.Image, processor, device: torch.device | None = None
) -> torch.Tensor:
    """Same Resize+CenterCrop convention as features.py, at CLIP's native
    224x224, using CLIP's own mean/std -- keeps map_point_to_model_space's
    coordinate math exact for this model too."""
    device = device or get_device()
    preprocess = _build_preprocess(processor)
    img = img.convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


def extract_multiscale_features_clip(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    layers: list[int] = (2, 5, 8, 11),
    fusion: Literal["mean", "concat"] = "mean",
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """CLIP equivalent of features.extract_multiscale_features, via HF's
    output_hidden_states (same pattern as features_v3.py's DINOv3 path).

    CLIP has 1 CLS token, no register tokens.
    """
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    n_available = len(hidden_states) - 1
    for layer_idx in layers:
        if layer_idx >= n_available:
            raise ValueError(
                f"Requested layer {layer_idx} but this CLIP variant only has "
                f"{n_available} blocks (0-indexed, max {n_available - 1}). "
                "Adjust `layers` for this model size."
            )

    grid = GRID_SIZE_CLIP
    layer_feats: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        feat = hidden_states[layer_idx + 1][:, 1:, :]  # drop CLS token
        b, n, d = feat.shape
        expected = grid * grid
        if n != expected:
            raise ValueError(
                f"Got {n} patch tokens after dropping the CLS token, expected "
                f"{expected} ({grid}x{grid}). Check IMG_SIZE_CLIP/PATCH_SIZE_CLIP "
                "against this model variant's actual config."
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
