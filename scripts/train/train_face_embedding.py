"""Train FaceProjectionHead with triplet loss, on top of a FROZEN DINOv2 backbone.

Expects a dataset laid out as:
    dataset_root/
        person_1/
            photo1.jpg
            photo2.jpg
        person_2/
            photo1.jpg
            ...

This is the standard face-dataset folder convention (LFW-funneled, CelebA
reorganized by identity, or your own collected photos all follow this
pattern). Needs at least 2 photos per identity to form triplets, and works
much better with several dozen+ identities and multiple photos each --
a handful of people with 2 photos each will badly overfit.

Usage:
    python scripts/train/train_face_embedding.py \
        --dataset-root ./datasets/faces --epochs 30 --output ./face_head.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from geco.face import EMBEDDING_DIM, FaceEmbedder, FaceProjectionHead  # noqa: E402


def load_identity_folders(dataset_root: Path) -> dict[str, list[Path]]:
    identities = {}
    for person_dir in sorted(dataset_root.iterdir()):
        if not person_dir.is_dir():
            continue
        photos = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if len(photos) >= 2:
            identities[person_dir.name] = photos
    return identities


class PooledFeatureCache(Dataset):
    """Precomputes each photo's frozen-backbone pooled feature ONCE, so training epochs
    only rerun the small trainable head, not the DINOv2 forward pass, every step.
    """

    def __init__(self, embedder: FaceEmbedder, identities: dict[str, list[Path]]):
        self.features: list[torch.Tensor] = []
        self.identity_ids: list[int] = []
        self.identity_names = list(identities.keys())

        print(f"Precomputing frozen backbone features for {sum(len(v) for v in identities.values())} photos...")
        for identity_idx, name in enumerate(self.identity_names):
            for photo_path in identities[name]:
                img = Image.open(photo_path).convert("RGB")
                feat = embedder.compute_pooled_features(img)
                self.features.append(feat.cpu())
                self.identity_ids.append(identity_idx)

        # index lookup: identity_idx -> list of positions in self.features
        self.by_identity: dict[int, list[int]] = {}
        for pos, idx in enumerate(self.identity_ids):
            self.by_identity.setdefault(idx, []).append(pos)

    def __len__(self):
        return len(self.features)

    def sample_triplet(self, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Anchor + positive (same identity) + negative (different identity), random sampling."""
        anchor_identity = rng.choice(list(self.by_identity.keys()))
        anchor_pos, positive_pos = rng.sample(self.by_identity[anchor_identity], 2)

        negative_identity = rng.choice([i for i in self.by_identity if i != anchor_identity])
        negative_pos = rng.choice(self.by_identity[negative_identity])

        return self.features[anchor_pos], self.features[positive_pos], self.features[negative_pos]


def train(cache: PooledFeatureCache, head: FaceProjectionHead, epochs: int, steps_per_epoch: int, margin: float, lr: float, seed: int):
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    device = next(head.parameters()).device

    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            anchors, positives, negatives = [], [], []
            for _ in range(16):  # mini-batch of triplets
                a, p, n = cache.sample_triplet(rng)
                anchors.append(a)
                positives.append(p)
                negatives.append(n)

            anchor_feat = torch.stack(anchors).to(device)
            positive_feat = torch.stack(positives).to(device)
            negative_feat = torch.stack(negatives).to(device)

            anchor_emb = head(anchor_feat)
            positive_emb = head(positive_feat)
            negative_emb = head(negative_feat)

            # Triplet loss on cosine distance (embeddings are L2-normalized, so
            # 1 - cosine_similarity is a valid distance in [0, 2]).
            dist_pos = 1 - (anchor_emb * positive_emb).sum(dim=-1)
            dist_neg = 1 - (anchor_emb * negative_emb).sum(dim=-1)
            loss = F.relu(dist_pos - dist_neg + margin).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1}/{epochs}  triplet_loss={avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Folder-per-identity face dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin (cosine-distance units)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="./face_head.pt")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    identities = load_identity_folders(dataset_root)
    if len(identities) < 2:
        raise SystemExit(
            f"Found only {len(identities)} identities with >=2 photos each in {dataset_root}. "
            "Need at least 2 identities to form negative pairs."
        )
    print(f"Loaded {len(identities)} identities, {sum(len(v) for v in identities.values())} total photos.")

    print("Loading DINOv2 (frozen backbone)...")
    embedder = FaceEmbedder()

    cache = PooledFeatureCache(embedder, identities)

    print(f"\nTraining projection head for {args.epochs} epochs...")
    train(cache, embedder.head, args.epochs, args.steps_per_epoch, args.margin, args.lr, args.seed)

    torch.save(embedder.head.state_dict(), args.output)
    print(f"\nSaved trained head to {args.output}")
    print(
        "Load it later with:\n"
        "  head = FaceProjectionHead()\n"
        f"  head.load_state_dict(torch.load('{args.output}'))\n"
        "  embedder = FaceEmbedder(head=head)"
    )


if __name__ == "__main__":
    main()
