"""Face verification on top of DINOv2 -- a DIFFERENT task from the rest of this
package. Everything else here does CORRESPONDENCE (where is point X in image
B?). This module does VERIFICATION (are these two face photos the SAME
PERSON?) -- no keypoints, no Optimal Transport, just an embedding + a
similarity threshold, the standard face-recognition setup.

Design: DINOv2's backbone stays FROZEN (same as the rest of this project --
no full fine-tuning, which would need much more data/compute than is
realistic here). A small trainable projection head sits on top, trained with
a triplet loss so that same-identity photos land close together in embedding
space and different-identity photos land far apart. This mirrors how DINO
is actually used for face recognition in practice: frozen backbone + a
lightweight metric-learning head, not full backbone fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .features import get_device, load_dinov2, preprocess_image

EMBEDDING_DIM = 128  # final face embedding size, after the projection head


class FaceProjectionHead(nn.Module):
    """Small trainable head: pooled DINOv2 features -> a compact, L2-normalized face embedding."""

    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pooled_feat: torch.Tensor) -> torch.Tensor:
        embedding = self.net(pooled_feat)
        return F.normalize(embedding, dim=-1)  # unit-norm, so cosine similarity == dot product


class FaceEmbedder:
    """Combines a frozen DINOv2 backbone with a (trainable) projection head."""

    def __init__(self, head: FaceProjectionHead | None = None, device: torch.device | None = None):
        self.device = device or get_device()
        self.backbone = load_dinov2(device=self.device)
        self.head = (head or FaceProjectionHead()).to(self.device)

    def _global_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """DINOv2's own image-level embedding (the CLS token from a direct forward pass),
        NOT a mean-pool of extract_multiscale_features' patch grid.

        extract_multiscale_features() calls get_intermediate_layers(..., return_class_token=False)
        -- it only ever returns patch tokens, no CLS token exists in its output. Mean-pooling
        those patches averages away exactly the fine identity detail (eye/nose shape) that
        distinguishes faces, leaving mostly coarse, similar-across-photos signal (lighting,
        framing) -- that's what was causing the projection head to collapse to a single point
        (triplet_loss stuck exactly at the margin). Calling the backbone directly gives DINOv2's
        native global embedding, the standard way DINO is used for image-level tasks like
        identity/classification (as opposed to the patch-grid features the rest of this
        project correctly uses for spatial correspondence).
        """
        with torch.no_grad():
            return self.backbone(tensor)  # [1, 768] -- CLS token, DINOv2's pooled output

    def embed(self, img: Image.Image) -> torch.Tensor:
        """Returns a single [EMBEDDING_DIM] L2-normalized embedding for one face image."""
        tensor = preprocess_image(img, self.device)
        pooled = self._global_embedding(tensor)
        return self.head(pooled).squeeze(0)

    def embed_batch_from_tensors(self, pooled_feats: torch.Tensor) -> torch.Tensor:
        """For training: takes PRECOMPUTED global backbone embeddings [B, 768] (since the
        backbone is frozen, its output is fixed per-image and worth caching once rather
        than rerunning on every training step -- see train_face_embedding.py).
        """
        return self.head(pooled_feats)

    def compute_pooled_features(self, img: Image.Image) -> torch.Tensor:
        """The frozen-backbone half only -- used once per image during dataset prep to cache
        features, so training epochs only run the cheap trainable head repeatedly.
        """
        tensor = preprocess_image(img, self.device)
        return self._global_embedding(tensor).squeeze(0)  # [768]


def verify(embedder: FaceEmbedder, img_a: Image.Image, img_b: Image.Image, threshold: float = 0.5) -> tuple[bool, float]:
    """Returns (is_same_person, cosine_similarity). threshold should be calibrated on a
    held-out verification-pairs set (see evaluate_face_verification.py) -- 0.5 is a
    reasonable untuned starting point for L2-normalized embeddings, not a proven value.
    """
    emb_a = embedder.embed(img_a)
    emb_b = embedder.embed(img_b)
    similarity = torch.dot(emb_a, emb_b).item()  # both unit-norm -> dot product == cosine similarity
    return similarity >= threshold, similarity
