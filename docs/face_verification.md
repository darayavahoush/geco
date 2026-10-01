# Face Verification with DINOv2 + ArcFace Metric Head

This document details the face verification module within GECO, which verifies whether two face photos represent the same individual.

---

## 1. Architecture Design

Unlike semantic correspondence (which maps localized patch keypoints between images using patch tokens and Optimal Transport), face verification is an **identity classification and metric learning** task.

```
┌─────────────────┐      ┌─────────────────────────┐      ┌───────────────────────────────┐
│ Input Face (A)  │─────▶│ Canonical 2-Point       │─────▶│ Frozen Meta DINOv2 ViT-B/14   │
└─────────────────┘      │ Alignment (224×224)     │      │ (Global CLS Token, 768-d)     │
                         └─────────────────────────┘      └──────────────┬────────────────┘
                                                                         │
                                                                         ▼
                                                          ┌───────────────────────────────┐
                                                          │ ArcFace Projection Head       │
                                                          │ Linear(768, 256) -> ReLU ->   │
                                                          │ Linear(256, 128) -> L2 Normal.│
                                                          └──────────────┬────────────────┘
                                                                         │
                                                                         ▼
                                                          ┌───────────────────────────────┐
                                                          │ 128-d Hypersphere Embedding   │
                                                          └──────────────┬────────────────┘
                                                                         │
                                   Cosine Similarity: cos(θ) = u · v ───┴────▶  Verdict (τ = 0.30)
```

### Key Architectural Decisions:
1. **Frozen DINOv2 Backbone**:
   Full fine-tuning of an 86M parameter vision transformer on small/medium face datasets leads to severe overfitting and catastrophic forgetting. Freezing DINOv2 preserves rich, generalized visual representations.
2. **Global CLS Token vs. Mean-Pooling**:
   Mean-pooling patch tokens averages away fine identity details (eye/nose curvature, subtle contours), leading to representation collapse. Using the native CLS token preserves discriminative identity signal.
3. **ArcFace Angular Margin Loss**:
   Triplets loss ($L = \max(0, d(a,p) - d(a,n) + m)$) often suffers from unstable negative sampling and slow convergence. ArcFace applies an additive angular margin $m=0.3$ on normalized weights and embeddings:
   $$\mathcal{L} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \ne y_i} e^{s \cos \theta_j}}$$
   This forces compact intra-class variance and wide inter-class separation on the unit hypersphere.

---

## 2. Dataset Curation & Training

The model was trained on a combined corpus:
- **LFW (Labeled Faces in the Wild)**: 1,680 identities, 9,164 photos.
- **YouTube Faces (YTF)**: 500 identities, 8,221 aligned video frames extracted across distinct video clips with ground-truth 68-point landmarks (`prepare_ytf.py`).
- **Total Training Corpus**: 2,163 unique identities across 17,385 aligned images.

### Training Hyperparameters:
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$, Cosine Annealing to $10^{-5}$)
- **Scale Factor ($s$)**: $30.0$
- **Angular Margin ($m$)**: $0.3$
- **Batch Size**: $128$
- **Epochs**: $20$

---

## 3. Benchmark Verification Results

Evaluated on the standard 500-pair held-out verification protocol (`datasets/faces/pairs_large.txt`):

| Model / Configuration | Training Corpus | Best Threshold ($\tau$) | Verification Accuracy |
| :--- | :--- | :--- | :--- |
| **Baseline (Untrained Head)** | None | 0.00 | ~50.0% (Chance) |
| **Triplet Loss Head** (`face_head.pt`) | LFW Only | 0.55 | 89.60% |
| **ArcFace Head** (`face_head_arcface.pt`) | **LFW + YouTube Faces** | **0.30** | **99.40%** |

### Detailed Error Analysis (500 pairs):
- **True Positives**: 249 / 250 (99.6%)
- **True Negatives**: 248 / 250 (99.2%)
- **False Positives**: 2 / 250 (0.8%)
- **False Negatives**: 1 / 250 (0.4%)
- **Optimal Decision Threshold**: $\tau = 0.30$

---

## 4. Frontend & API Features

The web frontend includes:
1. **Interactive Benchmark Presets Shelf**: Instant one-click testing of pre-curated positive and negative pairs from LFW and YouTube Faces.
2. **In-Browser Webcam / Selfie Capture**: Capture live photos directly via browser camera with an alignment guide oval.
3. **Interactive Decision Threshold Slider**: Live slider ($0.00$ to $0.80$) with instant dynamic recalculation of verdicts and margins.
4. **Cosine Similarity Spectrum Meter**: Visual color-coded continuum highlighting the separation between match, ambiguous buffer, and mismatch zones.
5. **Dual Inspection Mode**: Toggle between original uploaded images and normalized 224×224 aligned crops passed to the neural network.
6. **Detailed Metric Vector Breakdown**: Cosine similarity, cosine distance, L2 Euclidean distance on unit sphere, and decision margin.
