"""Face verification on top of DINOv2 -- a DIFFERENT task from the rest of this
package. Everything else here does CORRESPONDENCE (where is point X in image
B?). This module does VERIFICATION (are these two face photos the SAME
PERSON?) -- no keypoints, no Optimal Transport, just an embedding + a
similarity threshold, the standard face-recognition setup.

Design: DINOv2's backbone stays FROZEN (same as the rest of this project --
no full fine-tuning, which would need much more data/compute than is
realistic here). A small trainable projection head sits on top, trained with
a triplet loss (or ArcFace-style classification loss -- see
scripts/train/train_face_arcface.py) so that same-identity photos land close
together in embedding space and different-identity photos land far apart.

Every photo is face-aligned (see face_align.py) before being embedded, so
training and inference always see the same kind of input -- consistency
between the two matters as much as the alignment itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .features import get_device, load_dinov2, preprocess_image
from .face_align import align_face

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
        NOT a mean-pool of extract_multiscale_features'' patch grid. See git history for
        why this matters (mean-pooling caused embedding collapse).
        """
        with torch.no_grad():
            return self.backbone(tensor)  # [1, 768] -- CLS token, DINOv2''s pooled output

    def embed_with_aligned(self, img: Image.Image) -> tuple[torch.Tensor, Image.Image]:
        """Returns (embedding, aligned_image). Aligned image is the 224x224 crop fed to DINOv2."""
        aligned = align_face(img)
        tensor = preprocess_image(aligned, self.device)
        pooled = self._global_embedding(tensor)
        return self.head(pooled).squeeze(0), aligned

    def embed(self, img: Image.Image) -> torch.Tensor:
        """Returns a single [EMBEDDING_DIM] L2-normalized embedding for one face image."""
        emb, _ = self.embed_with_aligned(img)
        return emb

    def embed_batch_from_tensors(self, pooled_feats: torch.Tensor) -> torch.Tensor:
        """For training: takes PRECOMPUTED global backbone embeddings [B, 768] (since the
        backbone is frozen, its output is fixed per-image and worth caching once rather
        than rerunning on every training step).
        """
        return self.head(pooled_feats)

    def compute_pooled_features(self, img: Image.Image) -> torch.Tensor:
        """The frozen-backbone half only -- used once per image during dataset prep to cache
        features, so training epochs only run the cheap trainable head repeatedly.
        """
        aligned = align_face(img)
        tensor = preprocess_image(aligned, self.device)
        return self._global_embedding(tensor).squeeze(0)  # [768]


def verify(embedder: FaceEmbedder, img_a: Image.Image, img_b: Image.Image, threshold: float = 0.3) -> tuple[bool, float]:
    """Returns (is_same_person, cosine_similarity). threshold should be calibrated on a
    held-out verification-pairs set (see evaluate_face_verification.py).
    """
    emb_a = embedder.embed(img_a)
    emb_b = embedder.embed(img_b)
    similarity = torch.dot(emb_a, emb_b).item()  # both unit-norm -> dot product == cosine similarity
    return similarity >= threshold, similarity


def verify_detailed(
    embedder: FaceEmbedder,
    img_a: Image.Image,
    img_b: Image.Image,
    threshold: float = 0.30,
) -> dict:
    """Returns a rich evaluation dictionary with cosine similarity, L2 Euclidean distance,
    calibrated verdict, confidence percentage, verdict category, and the aligned face crops.
    """
    emb_a, aligned_a = embedder.embed_with_aligned(img_a)
    emb_b, aligned_b = embedder.embed_with_aligned(img_b)

    similarity = float(torch.dot(emb_a, emb_b).item())
    is_same = similarity >= threshold
    margin = similarity - threshold

    # Distance metrics on unit sphere:
    # Cosine distance in [0, 2]
    cosine_distance = max(0.0, 1.0 - similarity)
    # Euclidean distance between unit vectors: ||u - v|| = sqrt(2 - 2 * cos(theta))
    euclidean_distance = float((2.0 - 2.0 * min(1.0, max(-1.0, similarity))) ** 0.5)

    # Category classification based on margin from decision boundary
    if abs(margin) < 0.08:
        verdict_category = "ambiguous"
    elif margin >= 0.25:
        verdict_category = "confident_match"
    elif margin > 0:
        verdict_category = "likely_match"
    elif margin <= -0.25:
        verdict_category = "confident_different"
    else:
        verdict_category = "likely_different"

    # Confidence percentage (50% on boundary, approaching 99.9% further away)
    scaled_dist = min(1.0, abs(margin) / 0.40)
    confidence = round(min(99.9, 50.0 + scaled_dist * 49.9), 1)

    return {
        "is_same_person": is_same,
        "similarity": similarity,
        "threshold": threshold,
        "margin": margin,
        "confidence": confidence,
        "verdict_category": verdict_category,
        "cosine_distance": cosine_distance,
        "euclidean_distance": euclidean_distance,
        "aligned_a": aligned_a,
        "aligned_b": aligned_b,
    }
