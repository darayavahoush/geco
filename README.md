---
title: GECO
emoji: 🐱
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# GECO-Enhanced — Confidence-Aware Semantic Correspondence

Find semantic keypoint correspondences between two images of the same object
category (two different birds, two different chairs, the same object from
two viewpoints) using DINOv2 features and confidence-weighted Optimal
Transport. Click a point on one image, see where it maps to on the other.

This is an enhanced version of **GECO**-style matching: on top of the
standard "extract features → cosine similarity → Sinkhorn OT" pipeline, this
project adds a *confidence* signal that:

1. re-weights the OT marginal distributions (distinctive patches get more
   mass than generic/background patches),
2. sets a **per-patch adaptive "dustbin" threshold**, so low-confidence
   patches can be discarded ("no match") more easily instead of being forced
   into a bad correspondence, and
3. directly scales the cosine similarity cost matrix before matching.

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│  Source img  │────▶│  DINOv2 features  │────▶│                        │
└──────────────┘     │  (multi-scale,    │     │   Confidence-weighted  │
┌──────────────┐     │  layers 2,5,8,11) │────▶│   Sinkhorn OT matching │──▶ transport plan
│  Target img  │────▶│                   │     │                        │
└──────────────┘     └──────────────────┘     └────────────────────────┘
                              │
                              ▼
                  per-patch confidence (v1: entropy / v2: distance-from-mean)
                              │
                              ▼
                confidence-aware marginals + adaptive dustbin
```

## Project structure

```
geco/
├── src/geco/          # Core pipeline package (framework-agnostic)
│   ├── features.py     # DINOv2 loading, preprocessing, multi-scale extraction
│   ├── confidence.py    # Patch confidence estimation + adaptive dustbin
│   ├── matching.py      # Confidence-aware marginals + Sinkhorn OT matching
│   ├── keypoints.py      # Pixel-level keypoint transfer
│   └── visualize.py      # PCA/heatmap rendering to base64 PNGs
├── backend/            # FastAPI service wrapping the package for the frontend
│   └── main.py
├── frontend/           # React (Vite) UI
│   └── src/
├── scripts/
│   └── download_datasets.sh   # optional: PF-PASCAL / SPair-71k / CUB-200 for eval
├── docker-compose.yml
└── pyproject.toml
```

## Quick start (local, no Docker)

**Backend**

```bash
cd geco
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                                        # installs src/geco
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

First request will download DINOv2 ViT-B/14 weights (~330MB) via `torch.hub`
and cache them in `~/.cache/torch/hub`. A GPU is strongly recommended but not
required — `geco.features.get_device()` falls back to CPU/MPS automatically.

**Frontend**

```bash
cd geco/frontend
cp .env.example .env      # points at http://localhost:8000 by default
npm install
npm run dev
```

Open http://localhost:5173, upload two images, tune the sliders, hit **Run
matching**, then click anywhere on the source image to see the matched point
on the target.

## Quick start (Docker)

```bash
docker compose up --build
```

Backend on `:8000`, frontend dev server on `:5173`. For a GPU, uncomment the
`deploy.resources` block in `docker-compose.yml` (requires the NVIDIA
Container Toolkit).

## API

| Method | Path                        | Description                                             |
|--------|-----------------------------|-----------------------------------------------------------|
| GET    | `/health`                   | Liveness + model/device status                          |
| POST   | `/api/match`                 | Upload `src_image`, `trg_image` + params → confidence/PCA maps + `session_id` |
| POST   | `/api/keypoint`               | `{session_id, pixel_x, pixel_y}` → matched pixel on target |
| GET    | `/api/match-heatmap/{id}`      | Debug: full transport heatmap for one source patch        |

Interactive docs at `http://localhost:8000/docs` (FastAPI's built-in Swagger UI).

## Pipeline parameters (exposed as sliders in the UI)

- **alpha** (0–1): blend between uniform and confidence-weighted OT marginals.
- **z_base**: baseline "no-match" cost for a maximally confident patch.
- **z_range**: how much a low-confidence patch's dustbin threshold is lowered.
- **reg**: Sinkhorn entropic regularization (lower = sharper but slower to converge).

## Reproducing benchmark evaluation

```bash
./scripts/download_datasets.sh ./datasets
```

Downloads PF-PASCAL, SPair-71k, and CUB-200-2011 into `./datasets/` for
correspondence benchmark evaluation. Not required to run the demo app.

## Notes on the original notebook

This repo is a refactor of an exploratory Colab notebook: the core algorithm
(multi-scale DINOv2 features, entropy/distance-based confidence, confidence-
weighted marginals, adaptive dustbin, Sinkhorn matching) is unchanged from the
best-performing version (`geco_enhanced_match_v3`), just reorganized into a
tested, importable package instead of sequential notebook cells with several
duplicate/overwritten function definitions.

## License

MIT — see [LICENSE](LICENSE).
