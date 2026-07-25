"""DINOv2 model loading, image preprocessing, and multi-scale feature extraction."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# DINOv2 ViT-B/14 uses 14x14 patches; 518 = 37 * 14, giving a clean 37x37 patch grid.
IMG_SIZE = 518
PATCH_SIZE = 14
GRID_SIZE = IMG_SIZE // PATCH_SIZE  # 37

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_preprocess = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),  # scales shorter side to IMG_SIZE, preserving aspect ratio
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dinov2(variant: str = "dinov2_vitb14", device: torch.device | None = None) -> torch.nn.Module:
    """Load a frozen, eval-mode DINOv2 backbone from torch.hub.

    Args:
        variant: one of dinov2_vits14 / vitb14 / vitl14 / vitg14.
        device: target device; auto-detected if omitted.
    """
    device = device or get_device()
    model = torch.hub.load("facebookresearch/dinov2", variant)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def preprocess_image(img: Image.Image, device: torch.device | None = None) -> torch.Tensor:
    """Convert a PIL image into a normalized [1, 3, 518, 518] tensor on the given device."""
    device = device or get_device()
    img = img.convert("RGB")
    tensor = _preprocess(img).unsqueeze(0).to(device)
    return tensor


def map_point_to_model_space(
    x: float, y: float, orig_width: int, orig_height: int, image_size: int = IMG_SIZE
) -> tuple[float, float] | None:
    """Map a pixel coordinate in an original image to this model's resize+center-crop space.

    Mirrors the Resize(shorter side to image_size) + CenterCrop(image_size)
    pipeline in `_preprocess`, so ground-truth keypoints from a benchmark
    dataset land in the same coordinate space as the model's patch grid.

    Returns None if the point falls outside the center crop (i.e. it was in
    a part of the image discarded by cropping) — evaluation code should skip
    such points rather than silently score against a garbage location.
    """
    scale = image_size / min(orig_width, orig_height)
    new_w, new_h = orig_width * scale, orig_height * scale
    crop_x0 = (new_w - image_size) / 2
    crop_y0 = (new_h - image_size) / 2

    mapped_x = x * scale - crop_x0
    mapped_y = y * scale - crop_y0

    if not (0 <= mapped_x < image_size and 0 <= mapped_y < image_size):
        return None
    return mapped_x, mapped_y


def extract_multiscale_features(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    layers: list[int] = (2, 5, 8, 11),
    fusion: Literal["mean", "concat"] = "mean",
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Extract and fuse features from several DINOv2 transformer blocks.

    Fusing mid+late layers gives more spatially precise, less semantically
    "washed out" features than using only the final layer.

    Returns:
        fused: [1, D, H, W] (D = embed_dim, or embed_dim * len(layers) if fusion='concat')
        layer_feats: dict mapping layer index -> [1, D, H, W] feature map, for inspection.
    """
    with torch.no_grad():
        outputs = model.get_intermediate_layers(
            img_tensor, n=list(layers), return_class_token=False
        )

    grid = GRID_SIZE
    layer_feats: dict[int, torch.Tensor] = {}
    for layer_idx, feat in zip(layers, outputs):
        # feat: [1, H*W, D] -> [1, D, H, W]
        b, n, d = feat.shape
        feat_map = feat.transpose(-1, -2).reshape(b, d, grid, grid)
        layer_feats[layer_idx] = feat_map

    stacked = torch.stack(list(layer_feats.values()), dim=0)  # [L, 1, D, H, W]
    if fusion == "mean":
        fused = stacked.mean(dim=0)
    elif fusion == "concat":
        fused = torch.cat(list(layer_feats.values()), dim=1)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion!r}")

    return fused, layer_feats
