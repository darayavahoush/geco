"""ArcFace training of the face projection head that is ROBUST to glasses and beards.

Same recipe as train_face_arcface.py (frozen DINOv2 backbone, small trainable head,
ArcFace margin loss), with one change: every training photo contributes several
views that all share the photo's identity label --
    clean, + synthetic glasses, + synthetic beard, + both
(see geco.appearance_aug). The head therefore learns that adding/removing those
features must not move a person's embedding. Real photos that already show glasses or
beards (CelebA, see scripts/data/prepare_celeba.py) are used as they are, on top.

Backbone features are cached, so the expensive pass (DINOv2 over N photos x views) runs
once; re-running with --cache-features reuses it and trains in seconds.

Usage:
  python scripts/train/train_face_arcface_robust.py \
      --dataset-root datasets/faces/lfw datasets/faces/ytf_frames datasets/faces/celeba_train \
      --views clean glasses beard both --cache-features cache/robust_feats.pt \
      --output face_head_robust.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geco.appearance_aug import random_appearance_change  # noqa: E402
from geco.face import EMBEDDING_DIM, FaceEmbedder  # noqa: E402
from geco.face_align import extract_face_bbox_and_crop  # noqa: E402
from train_face_arcface import ArcMarginProduct, load_dataset  # noqa: E402


def precompute(embedder, samples, views, seed, max_per_identity):
    rng = np.random.default_rng(seed)
    feats, labels, tags = [], [], []
    per_id: dict[int, int] = {}
    skipped = 0
    for i, (path, label) in enumerate(samples):
        if max_per_identity and per_id.get(label, 0) >= max_per_identity:
            continue
        if (i + 1) % 200 == 0 or i == 0 or i + 1 == len(samples):
            print(f"  [{i + 1}/{len(samples)}] photos", flush=True)
        try:
            crop, meta = extract_face_bbox_and_crop(Image.open(path).convert("RGB"))
            if meta.get("method") != "yunet_dnn":      # don't train on guessed crops
                skipped += 1
                continue
            for v in views:
                img = crop if v == "clean" else random_appearance_change(crop, rng, mode=v)[0]
                feats.append(embedder.pooled_from_aligned(img).cpu())
                labels.append(label)
                tags.append(v)
            per_id[label] = per_id.get(label, 0) + 1
        except Exception:
            skipped += 1
    print(f"skipped {skipped} photos (no face detected / unreadable)")
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path, nargs="+")
    ap.add_argument("--views", nargs="+", default=["clean", "glasses", "beard", "both"], choices=["clean", "glasses", "beard", "both"])
    ap.add_argument("--max-per-identity", type=int, default=0, help="cap photos per identity (0 = all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-features", type=Path, default=None)
    ap.add_argument("--init-head", type=Path, default=None, help="start from an existing head, e.g. face_head_arcface.pt")
    ap.add_argument("--output", type=Path, default=Path("face_head_robust.pt"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    if a.cache_features and a.cache_features.exists():
        blob = torch.load(a.cache_features, map_location="cpu")
        feats, labels, num_classes = blob["feats"], blob["labels"], blob["num_classes"]
        print(f"loaded cache: {feats.shape[0]} views, {num_classes} identities", flush=True)
    else:
        samples, num_classes = load_dataset(a.dataset_root)
        print(f"{len(samples)} photos, {num_classes} identities, views={a.views}", flush=True)
        embedder = FaceEmbedder(device=device)
        feats, labels, _ = precompute(embedder, samples, a.views, a.seed, a.max_per_identity)
        if a.cache_features:
            a.cache_features.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"feats": feats, "labels": labels, "num_classes": num_classes}, a.cache_features)
            print("cached ->", a.cache_features, flush=True)
        del embedder

    embedder = FaceEmbedder(device=device)
    if a.init_head:
        embedder.head.load_state_dict(torch.load(a.init_head, map_location=device))
        print("initialised from", a.init_head)
    arc = ArcMarginProduct(EMBEDDING_DIM, num_classes, s=a.scale, m=a.margin).to(device)
    opt = torch.optim.Adam(list(embedder.head.parameters()) + list(arc.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)

    feats, labels = feats.to(device), labels.to(device)
    n = feats.shape[0]
    for epoch in range(1, a.epochs + 1):
        embedder.head.train(); arc.train()
        perm = torch.randperm(n, device=device)
        tot = correct = 0
        loss_sum = 0.0
        for s in range(0, n, a.batch_size):
            idx = perm[s:s + a.batch_size]
            logits = arc(embedder.head(feats[idx]), labels[idx])
            loss = F.cross_entropy(logits, labels[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(idx); correct += (logits.argmax(1) == labels[idx]).sum().item(); tot += len(idx)
        sched.step()
        if epoch % 2 == 0 or epoch == a.epochs:
            print(f"  epoch {epoch:02d}/{a.epochs}  loss={loss_sum / tot:.4f}  acc={100 * correct / tot:.2f}%", flush=True)

    embedder.head.eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedder.head.state_dict(), a.output)
    print("saved", a.output)


if __name__ == "__main__":
    main()
