"""Train a lightweight keypoint regressor on top of frozen DINOv2 features,
comparing two label sources:

  1. AUTO-LABELED: a handful of hand-annotated "seed" images propagate labels
     onto a much larger pool via Component 2A (geco.propagation), and the
     regressor trains on those pseudo-labels.
  2. REAL-LABELED: an equal-size set of images using their real CUB
     annotations directly (the "if we'd hand-labeled this many instead"
     upper-bound comparison).

Both are evaluated on the same held-out real-labeled test set. The headline
result for the project write-up: how close does (1) get to (2) despite
costing far fewer human annotation hours — i.e. is propagation actually
saving useful annotation effort, not just producing noisy labels.

Usage:
    python scripts/train/train_keypoint_regressor.py \
        --cub-root ./datasets/cub/CUB_200_2011 \
        --n-seed-images 20 \
        --n-pool-images 300 \
        --n-test-images 100 \
        --epochs 40
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.cub import CubImage, load_cub  # noqa: E402
from geco.features import (  # noqa: E402
    extract_multiscale_features,
    get_device,
    load_dinov2,
    map_point_to_model_space,
    preprocess_image,
)
from geco.propagation import SeedAnnotation, propagate_keypoints  # noqa: E402

# CUB-200-2011's 15 standard part names, fixed order so the regressor has a stable output layout.
PART_NAMES = [
    "back", "beak", "belly", "breast", "crown", "forehead",
    "left_eye", "left_leg", "left_wing", "nape", "right_eye",
    "right_leg", "right_wing", "tail", "throat",
]


class KeypointRegressor(nn.Module):
    """Global-average-pooled DINOv2 feature -> per-part (x, y) in normalized [0,1] model space."""

    def __init__(self, in_dim: int = 768, hidden: int = 256, n_parts: int = len(PART_NAMES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_parts * 2),
        )
        self.n_parts = n_parts

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).reshape(-1, self.n_parts, 2)


class KeypointFeatureDataset(Dataset):
    """Precomputed (feature_vector, target_coords, mask) triples — features cached once, not recomputed per epoch."""

    def __init__(self, features: list[torch.Tensor], labels: list[dict[str, tuple[float, float]]]):
        self.features = features
        self.targets = []
        self.masks = []
        for label in labels:
            target = torch.zeros(len(PART_NAMES), 2)
            mask = torch.zeros(len(PART_NAMES))
            for i, name in enumerate(PART_NAMES):
                if name in label:
                    x, y = label[name]
                    target[i] = torch.tensor([x / 518.0, y / 518.0])
                    mask[i] = 1.0
            self.targets.append(target)
            self.masks.append(mask)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.masks[idx]


def compute_global_feature(model: torch.nn.Module, img: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = preprocess_image(img, device)
    fused, _ = extract_multiscale_features(model, tensor)  # [1, D, H, W]
    return fused.mean(dim=(2, 3)).squeeze(0).detach().cpu()  # [D]


def real_labels_in_model_space(cub_img: CubImage, img: Image.Image) -> dict[str, tuple[float, float]]:
    mapped = {}
    for name, (x, y) in cub_img.keypoints.items():
        pt = map_point_to_model_space(x, y, img.width, img.height)
        if pt is not None:
            mapped[name] = pt
    return mapped


def train_regressor(dataset: KeypointFeatureDataset, epochs: int, lr: float = 1e-3) -> KeypointRegressor:
    model = KeypointRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for feats, targets, masks in loader:
            optimizer.zero_grad()
            preds = model(feats)
            mask = masks.unsqueeze(-1)  # [B, P, 1]
            loss = ((preds - targets) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.5f}")
    model.eval()
    return model


def evaluate_regressor(
    model: KeypointRegressor,
    test_features: list[torch.Tensor],
    test_cub_imgs: list[CubImage],
    test_pil_imgs: list[Image.Image],
    alpha_thresh: float,
) -> float:
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for feat, cub_img, pil_img in zip(test_features, test_cub_imgs, test_pil_imgs):
            pred = model(feat.unsqueeze(0)).squeeze(0) * 518.0  # [P, 2], back to model-pixel space
            bbox_scale = max(cub_img.bbox[2], cub_img.bbox[3])
            scale = 518 / min(pil_img.width, pil_img.height)
            threshold = alpha_thresh * bbox_scale * scale

            for i, name in enumerate(PART_NAMES):
                if name not in cub_img.keypoints:
                    continue
                gt = map_point_to_model_space(*cub_img.keypoints[name], pil_img.width, pil_img.height)
                if gt is None:
                    continue
                err = ((pred[i, 0].item() - gt[0]) ** 2 + (pred[i, 1].item() - gt[1]) ** 2) ** 0.5
                n_total += 1
                if err <= threshold:
                    n_correct += 1
    return n_correct / n_total if n_total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--n-seed-images", type=int, default=20, help="Hand-labeled seeds used for auto-propagation")
    parser.add_argument("--n-pool-images", type=int, default=300, help="Unlabeled pool auto-labeled via propagation")
    parser.add_argument("--n-test-images", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("Loading CUB-200-2011 annotations...")
    images = list(load_cub(args.cub_root).values())
    rng.shuffle(images)

    n_needed = args.n_seed_images + args.n_pool_images + args.n_test_images
    if len(images) < n_needed:
        raise SystemExit(f"Need {n_needed} images but only found {len(images)} in the dataset.")

    seed_cub = images[: args.n_seed_images]
    pool_cub = images[args.n_seed_images : args.n_seed_images + args.n_pool_images]
    test_cub = images[args.n_seed_images + args.n_pool_images : n_needed]

    print("Loading DINOv2...")
    model_backbone = load_dinov2()
    device = get_device()

    def load_pil(cub_imgs):
        return [Image.open(c.path).convert("RGB") for c in cub_imgs]

    print(f"Loading {len(seed_cub)} seed / {len(pool_cub)} pool / {len(test_cub)} test images...")
    seed_pil = load_pil(seed_cub)
    pool_pil = load_pil(pool_cub)
    test_pil = load_pil(test_cub)

    # ── Auto-labeling: propagate seed annotations onto the pool ────────────
    print("Building seed annotations...")
    seeds = [
        SeedAnnotation(image=img, keypoints={
            name: (round(pt[0]), round(pt[1]))
            for name, pt in real_labels_in_model_space(c, img).items()
        })
        for c, img in zip(seed_cub, seed_pil)
    ]
    seeds = [s for s in seeds if s.keypoints]

    print(f"Auto-labeling {len(pool_pil)} pool images via propagation (this is the slow part)...")
    auto_labels = []
    for i, pool_img in enumerate(pool_pil):
        results = propagate_keypoints(model_backbone, seeds, pool_img)
        auto_labels.append({r.name: r.pixel for r in results if r.accepted})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(pool_pil)}")

    real_pool_labels = [real_labels_in_model_space(c, img) for c, img in zip(pool_cub, pool_pil)]

    # ── Precompute features once (shared across both training runs) ────────
    print("Computing global features for pool + test images...")
    pool_features = [compute_global_feature(model_backbone, img, device) for img in pool_pil]
    test_features = [compute_global_feature(model_backbone, img, device) for img in test_pil]

    # ── Train on auto-labels ────────────────────────────────────────────────
    print("\nTraining regressor on AUTO-PROPAGATED pseudo-labels...")
    auto_dataset = KeypointFeatureDataset(pool_features, auto_labels)
    auto_model = train_regressor(auto_dataset, args.epochs)
    auto_pck = evaluate_regressor(auto_model, test_features, test_cub, test_pil, args.alpha)

    # ── Train on real labels (same pool images, ground-truth labels instead) ─
    print("\nTraining regressor on REAL ground-truth labels (upper-bound comparison)...")
    real_dataset = KeypointFeatureDataset(pool_features, real_pool_labels)
    real_model = train_regressor(real_dataset, args.epochs)
    real_pck = evaluate_regressor(real_model, test_features, test_cub, test_pil, args.alpha)

    print("\n" + "=" * 60)
    print(f"{'Training labels':<30}{'Test PCK@' + str(args.alpha):<15}")
    print("-" * 60)
    print(f"{'Auto-propagated (' + str(len(seeds)) + ' seeds)':<30}{auto_pck:<15.4f}")
    print(f"{'Real ground-truth':<30}{real_pck:<15.4f}")
    print("=" * 60)
    gap = real_pck - auto_pck
    print(f"Gap to real-label upper bound: {gap:+.4f}")
    print(
        f"Annotation cost: {len(seeds)} hand-labeled images -> {len(pool_pil)} auto-labeled, "
        f"vs. {len(pool_pil)} hand-labeled images for the real-label run."
    )


if __name__ == "__main__":
    main()
