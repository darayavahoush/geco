"""DINOv3 feature extraction, matching the interface geco.cross_model already
expects (load_dinov3, preprocess_image_v3, extract_multiscale_features_v3,
IMG_SIZE_V3, GRID_SIZE_V3).

DINOv3 is distributed on Hugging Face as a GATED repo -- you need:
  1. Requested + been granted access on the model's HF page
  2. Run `huggingface-cli login` (or set HF_TOKEN env var) locally so
     transformers can authenticate the download

IMPORTANT -- verify before running: I don't have your exact granted repo ID
(couldn't check it from this environment). The default below
("facebook/dinov3-vitb16-pretrain-lvd1689m") is my best-effort guess at
Meta's naming convention -- check the exact string on the model page you were
approved for and pass it via `load_dinov3(model_id=...)` if it differs.

Patch size / register tokens: DINOv3 (like DINOv2-with-registers) may prepend
a CLS token plus several register tokens before the patch tokens. Rather than
hardcoding an assumed count (risky to get wrong silently), the special-token
count is computed dynamically as `sequence_length - grid_size**2` at
extraction time -- robust to however many special tokens the checkpoint uses,
as long as GRID_SIZE_V3 correctly matches IMG_SIZE_V3 / PATCH_SIZE_V3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# DINOv3 ViT-B/16: 16x16 patches. 512 = 32*16, giving a clean 32x32 grid.
# VERIFY: if your granted checkpoint uses a different patch size (e.g. /14),
# update PATCH_SIZE_V3 and IMG_SIZE_V3 together so IMG_SIZE_V3 % PATCH_SIZE_V3 == 0.
IMG_SIZE_V3 = 512
PATCH_SIZE_V3 = 16
GRID_SIZE_V3 = IMG_SIZE_V3 // PATCH_SIZE_V3  # 32

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_preprocess_v3 = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE_V3),
        transforms.CenterCrop(IMG_SIZE_V3),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)

DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # VERIFY against your access grant


def load_dinov3(model_id: str = DEFAULT_DINOV3_MODEL_ID, device: torch.device | None = None):
    """Load a frozen, eval-mode DINOv3 backbone + its HF image processor.

    Requires: `pip install transformers`, `huggingface-cli login` (or HF_TOKEN
    env var) with approved access to the gated model_id.

    Returns (model, processor) -- the processor is accepted by
    preprocess_image_v3 but this module does its own resize/normalize
    (matching the rest of this codebase's pattern) rather than fully
    delegating to the processor, for consistent, predictable coordinate math.
    """
    from transformers import AutoImageProcessor, AutoModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def preprocess_image_v3(img: Image.Image, processor, device: torch.device | None = None) -> torch.Tensor:
    """Convert a PIL image into a [1, 3, 512, 512] tensor.

    `processor` is accepted for interface consistency with load_dinov3's
    return value but isn't used for resizing here -- we do our own
    resize+center-crop (same pattern as features.py's DINOv2 preprocessing)
    so IMG_SIZE_V3 / GRID_SIZE_V3 stay guaranteed-consistent regardless of
    whatever default size the HF processor might otherwise pick.
    """
    del processor  # accepted for signature compatibility, intentionally unused
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.convert("RGB")
    tensor = _preprocess_v3(img).unsqueeze(0).to(device)
    return tensor


def extract_multiscale_features_v3(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    layers: list[int] = (2, 5, 8, 11),
    fusion: str = "mean",
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Extract and fuse features from several DINOv3 transformer layers.

    Uses `output_hidden_states=True` (standard transformers API) rather than
    a DINOv2-style `get_intermediate_layers` method, since HF's AutoModel
    wrapper exposes hidden states uniformly across model families.
    hidden_states[0] is the embedding output; hidden_states[i] for i>=1 is
    the output after transformer block i.
    """
    with torch.no_grad():
        outputs = model(img_tensor, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    grid = GRID_SIZE_V3
    n_patches = grid * grid
    layer_feats: dict[int, torch.Tensor] = {}

    for layer_idx in layers:
        hs = hidden_states[layer_idx]  # [1, seq_len, D]
        seq_len = hs.shape[1]
        n_special = seq_len - n_patches  # CLS (+ any register tokens), computed dynamically
        if n_special < 0:
            raise ValueError(
                f"Layer {layer_idx}: sequence length {seq_len} is shorter than the expected "
                f"{n_patches} patch tokens ({grid}x{grid}). IMG_SIZE_V3/PATCH_SIZE_V3/GRID_SIZE_V3 "
                "likely don't match this checkpoint -- verify against the model's actual config."
            )
        patch_tokens = hs[:, n_special:, :]  # assumes special tokens are PREPENDED, standard ViT convention
        b, n, d = patch_tokens.shape
        feat_map = patch_tokens.transpose(-1, -2).reshape(b, d, grid, grid)
        layer_feats[layer_idx] = feat_map

    stacked = torch.stack(list(layer_feats.values()), dim=0)
    if fusion == "mean":
        fused = stacked.mean(dim=0)
    elif fusion == "concat":
        fused = torch.cat(list(layer_feats.values()), dim=1)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion!r}")

    return fused, layer_feats
