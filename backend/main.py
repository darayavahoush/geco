"""FastAPI backend for the GECO-Enhanced semantic correspondence demo.

Endpoints
---------
GET  /health                 liveness + model/device status
POST /api/match               upload two images -> confidence maps, PCA maps, match stats
POST /api/keypoint            given a match session + clicked pixel -> matched pixel on target

The `/api/match` endpoint keeps the resulting MatchResult in an in-memory
session store (keyed by a session id) so that `/api/keypoint` can be called
repeatedly for different clicks without recomputing features + OT each time.
For a class project demo this in-memory store is fine; swap for redis/db if
you need multi-worker deployment.
"""

from __future__ import annotations

import base64
import io
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geco.features import get_device, load_dinov2, IMG_SIZE  # noqa: E402
from geco.keypoints import transfer_keypoint  # noqa: E402
from geco.matching import MatchResult, geco_match  # noqa: E402
from geco.visualize import (  # noqa: E402
    render_anomaly_heatmap,
    render_confidence_heatmap,
    render_match_heatmap,
    render_pca_map,
)
from geco.propagation import SeedAnnotation, propagate_keypoints  # noqa: E402
from geco.anomaly import detect_anomalies  # noqa: E402
from geco.face import FaceEmbedder, FaceProjectionHead, verify as face_verify, verify_detailed  # noqa: E402

app = FastAPI(title="GECO-Enhanced API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_device = None
_sessions: dict[str, MatchResult] = {}  # session_id -> MatchResult
_SESSION_TTL_SECONDS = 30 * 60
_session_timestamps: dict[str, float] = {}


def _get_model():
    global _model, _device
    if _model is None:
        _device = get_device()
        _model = load_dinov2(device=_device)
    return _model


_FACE_HEAD_ARCFACE_PATH = Path(__file__).resolve().parent.parent / "face_head_arcface.pt"
_FACE_HEAD_TRIPLET_PATH = Path(__file__).resolve().parent.parent / "face_head.pt"
_face_embedder: FaceEmbedder | None = None
_face_threshold: float = 0.30
_face_model_name: str = "DINOv2 ViT-B/14 + ArcFace Head (LFW + YTF 500 IDs, 99.4% accuracy)"


def _get_face_embedder() -> tuple[FaceEmbedder, float, str]:
    """Lazily loads the trained face projection head. Prioritizes the ArcFace head
    (99.40% accuracy, threshold 0.30) if present, and falls back to the triplet
    loss head (89.60% accuracy, threshold 0.55).
    """
    global _face_embedder, _face_threshold, _face_model_name
    if _face_embedder is None:
        head = FaceProjectionHead()
        if _FACE_HEAD_ARCFACE_PATH.exists():
            head.load_state_dict(torch.load(_FACE_HEAD_ARCFACE_PATH, map_location="cpu"))
            _face_threshold = 0.30
            _face_model_name = "DINOv2 ViT-B/14 + ArcFace Head (LFW + YTF 500 IDs, 99.4% accuracy)"
        elif _FACE_HEAD_TRIPLET_PATH.exists():
            head.load_state_dict(torch.load(_FACE_HEAD_TRIPLET_PATH, map_location="cpu"))
            _face_threshold = 0.55
            _face_model_name = "DINOv2 ViT-B/14 + Triplet Head (LFW only, 89.6% accuracy)"
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Face projection head weights not found (expected {_FACE_HEAD_ARCFACE_PATH} or {_FACE_HEAD_TRIPLET_PATH}).",
            )
        _face_embedder = FaceEmbedder(head=head)
    return _face_embedder, _face_threshold, _face_model_name


def _prune_sessions() -> None:
    now = time.time()
    expired = [sid for sid, ts in _session_timestamps.items() if now - ts > _SESSION_TTL_SECONDS]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_timestamps.pop(sid, None)


class MatchResponse(BaseModel):
    session_id: str
    grid_size: int
    entropy: float
    mean_confidence: float
    src_confidence_png: str
    trg_confidence_png: str
    src_pca_png: str
    trg_pca_png: str


class KeypointRequest(BaseModel):
    session_id: str
    pixel_x: int
    pixel_y: int
    image_size: int = 518


class KeypointResponse(BaseModel):
    trg_pixel_x: int
    trg_pixel_y: int
    confidence: float
    is_dustbin: bool


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(_device) if _device else "not loaded yet",
        "model_loaded": _model is not None,
    }


@app.post("/api/match", response_model=MatchResponse)
async def match_images(
    src_image: UploadFile = File(...),
    trg_image: UploadFile = File(...),
    alpha: float = Form(0.8),
    z_base: float = Form(0.3),
    z_range: float = Form(0.25),
    reg: float = Form(0.05),
):
    _prune_sessions()
    model = _get_model()

    try:
        img_src = Image.open(io.BytesIO(await src_image.read())).convert("RGB")
        img_trg = Image.open(io.BytesIO(await trg_image.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded images: {exc}")

    result = geco_match(
        model, img_src, img_trg, alpha=alpha, z_base=z_base, z_range=z_range, reg=reg
    )

    # Recompute the feature maps once more for PCA rendering (cheap relative to matching,
    # keeps geco_match's return type lean).
    from geco.features import extract_multiscale_features, preprocess_image  # local import

    device = next(model.parameters()).device
    src_feat, _ = extract_multiscale_features(model, preprocess_image(img_src, device))
    trg_feat, _ = extract_multiscale_features(model, preprocess_image(img_trg, device))

    session_id = str(uuid.uuid4())
    _sessions[session_id] = result
    _session_timestamps[session_id] = time.time()

    return MatchResponse(
        session_id=session_id,
        grid_size=result.grid_size,
        entropy=result.entropy,
        mean_confidence=result.mean_confidence,
        src_confidence_png=render_confidence_heatmap(result.conf_src, "Source confidence"),
        trg_confidence_png=render_confidence_heatmap(result.conf_trg, "Target confidence"),
        src_pca_png=render_pca_map(src_feat, "Source features (PCA)"),
        trg_pca_png=render_pca_map(trg_feat, "Target features (PCA)"),
    )


@app.post("/api/keypoint", response_model=KeypointResponse)
def keypoint(req: KeypointRequest):
    result = _sessions.get(req.session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session expired or not found. Re-run matching.")

    match = transfer_keypoint(result, req.pixel_x, req.pixel_y, image_size=req.image_size)
    return KeypointResponse(
        trg_pixel_x=match.trg_pixel[0],
        trg_pixel_y=match.trg_pixel[1],
        confidence=match.confidence,
        is_dustbin=match.is_dustbin,
    )


@app.get("/api/match-heatmap/{session_id}")
def match_heatmap(session_id: str, patch_x: int, patch_y: int):
    """Optional endpoint: full heatmap of where one source patch's mass lands (for debugging/demo)."""
    result = _sessions.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    grid = result.grid_size
    patch_idx = patch_y * grid + patch_x
    png = render_match_heatmap(
        result.transport_plan[patch_idx], grid, f"Patch ({patch_x},{patch_y}) -> target"
    )
    return {"heatmap_png": png}


# ── Component 2 / Track A — Sparse-to-Dense Annotation Propagation ──────────


class SeedKeypointIn(BaseModel):
    name: str
    x: int
    y: int


class PropagatedKeypointOut(BaseModel):
    name: str
    x: int
    y: int
    confidence: float
    cycle_error: float
    n_votes: int
    geometric_inlier: bool
    accepted: bool


@app.post("/api/propagate", response_model=list[PropagatedKeypointOut])
async def propagate(
    target_image: UploadFile = File(...),
    seed_images: list[UploadFile] = File(...),
    seed_keypoints_json: list[str] = Form(...),  # one JSON array of {name,x,y} per seed image
    confidence_threshold: float = Form(0.15),
    cycle_error_threshold: float = Form(40.0),
    geometric_residual_threshold: float = Form(25.0),
):
    """Propagate keypoints from N labeled seed images onto one unlabeled target image."""
    import json

    if len(seed_images) != len(seed_keypoints_json):
        raise HTTPException(
            status_code=400, detail="seed_images and seed_keypoints_json must be the same length."
        )

    model = _get_model()
    target_img = Image.open(io.BytesIO(await target_image.read())).convert("RGB")

    seeds = []
    for upload, kp_json in zip(seed_images, seed_keypoints_json):
        img = Image.open(io.BytesIO(await upload.read())).convert("RGB")
        kps = {kp["name"]: (kp["x"], kp["y"]) for kp in json.loads(kp_json)}
        seeds.append(SeedAnnotation(image=img, keypoints=kps))

    results = propagate_keypoints(
        model,
        seeds,
        target_img,
        confidence_threshold=confidence_threshold,
        cycle_error_threshold=cycle_error_threshold,
        geometric_residual_threshold=geometric_residual_threshold,
    )
    return [
        PropagatedKeypointOut(
            name=r.name,
            x=r.pixel[0],
            y=r.pixel[1],
            confidence=r.confidence,
            cycle_error=r.cycle_error,
            n_votes=r.n_votes,
            geometric_inlier=r.geometric_inlier,
            accepted=r.accepted,
        )
        for r in results
    ]


# ── Component 2 / Track B — Correspondence-Guided Anomaly Detection ─────────


class AnomalyResponse(BaseModel):
    anomaly_score: float
    anomaly_heatmap_png: str
    n_dustbin_patches: int
    grid_size: int
    n_references: int


@app.post("/api/anomaly", response_model=AnomalyResponse)
async def anomaly(
    test_image: UploadFile = File(...),
    reference_images: list[UploadFile] = File(...),
):
    """Score a test image for anomalies relative to one or more golden reference images."""
    model = _get_model()
    test_img = Image.open(io.BytesIO(await test_image.read())).convert("RGB")
    ref_imgs = [Image.open(io.BytesIO(await f.read())).convert("RGB") for f in reference_images]

    result = detect_anomalies(model, test_img, ref_imgs)

    return AnomalyResponse(
        anomaly_score=result.anomaly_score,
        anomaly_heatmap_png=render_anomaly_heatmap(result.anomaly_map),
        n_dustbin_patches=int(result.dustbin_mask.sum().item()),
        grid_size=result.match.grid_size,
        n_references=result.n_references,
    )


# ── Cross-Model Trust — DINOv2/DINOv1/CLIP agreement as a free uncertainty signal ──
# Curated to dinov2+dino1 by default: the category sweep showed adding CLIP as a
# third member actively DILUTES the ensemble (avg gap 0.232 -> 0.282 without it,
# win-rate 7/18 -> 9/18) rather than helping, despite naive intuition that more
# independent models = more signal.
_cross_model_registry: dict = {}  # lazily populated, {name: (match_fn, image_size)}


def _get_cross_model_registry(selected: list[str]):
    from geco.cross_model import geco_match_v2, geco_match_dino1, geco_match_clip

    missing = [m for m in selected if m not in _cross_model_registry]
    for name in missing:
        if name == "dinov2":
            m = _get_model()
            _cross_model_registry["dinov2"] = (lambda s, t, m=m: geco_match_v2(m, s, t), IMG_SIZE)
        elif name == "dino1":
            from geco.features_dino1 import load_dino1, IMG_SIZE_DINO1

            m = load_dino1()
            _cross_model_registry["dino1"] = (lambda s, t, m=m: geco_match_dino1(m, s, t), IMG_SIZE_DINO1)
        elif name == "clip":
            from geco.features_clip import load_clip, IMG_SIZE_CLIP

            m, p = load_clip()
            _cross_model_registry["clip"] = (
                lambda s, t, m=m, p=p: geco_match_clip(m, p, s, t),
                IMG_SIZE_CLIP,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model '{name}'")
    return {k: v for k, v in _cross_model_registry.items() if k in selected}


class CrossModelPredictionOut(BaseModel):
    model: str
    x: float
    y: float
    confidence: float
    is_dustbin: bool


class CrossModelMatchResponse(BaseModel):
    predictions: list[CrossModelPredictionOut]
    agreement_px: float
    centroid_x: float
    centroid_y: float
    trust: str  # "high" | "medium" | "low" -- rough bucketing, tune against your own image scale


@app.post("/api/cross-model-match", response_model=CrossModelMatchResponse)
async def cross_model_match(
    src_image: UploadFile = File(...),
    trg_image: UploadFile = File(...),
    pixel_x: float = Form(...),
    pixel_y: float = Form(...),
    models: str = Form("dinov2,dino1"),
):
    """Transfer one clicked point through several independent backbones and report
    their agreement as a free, training-free trust signal (no fine-tuning, no labels).

    pixel_x/pixel_y are in the ORIGINAL uploaded source image's pixel space (not any
    model's internal crop) -- send whatever coordinate the browser reports on the
    natural (unscaled) image, no need to know any model's internal input size.
    """
    from geco.cross_model import transfer_and_map, mean_pairwise_distance

    selected = [m.strip() for m in models.split(",") if m.strip()]
    if len(selected) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compute agreement.")

    registry = _get_cross_model_registry(selected)
    src_img = Image.open(io.BytesIO(await src_image.read())).convert("RGB")
    trg_img = Image.open(io.BytesIO(await trg_image.read())).convert("RGB")

    results = {name: fn(src_img, trg_img) for name, (fn, _) in registry.items()}

    predictions: list[CrossModelPredictionOut] = []
    points: list[tuple[float, float]] = []
    for name, (_, image_size) in registry.items():
        out = transfer_and_map(
            results[name], image_size, (pixel_x, pixel_y),
            src_img.width, src_img.height, trg_img.width, trg_img.height,
        )
        if out is None:
            continue  # point fell outside this model's crop
        (tx, ty), conf, is_dustbin = out
        predictions.append(CrossModelPredictionOut(model=name, x=tx, y=ty, confidence=conf, is_dustbin=is_dustbin))
        points.append((tx, ty))

    if len(points) < 2:
        raise HTTPException(
            status_code=422,
            detail="Fewer than 2 models produced a valid prediction (point may be outside crop for some models).",
        )

    agreement_px = mean_pairwise_distance(points)
    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)

    # NOTE: these thresholds are a rough starting point, not calibrated against a
    # labeled validation set. Before using "trust" as a real accept/reject gate,
    # calibrate them the same way propagation.py's confidence_threshold was tuned
    # -- via a coverage/accuracy sweep on held-out labeled pairs (see
    # scripts/eval/sweep_propagation_thresholds.py for the pattern to reuse).
    if agreement_px < 20:
        trust = "high"
    elif agreement_px < 60:
        trust = "medium"
    else:
        trust = "low"

    return CrossModelMatchResponse(
        predictions=predictions,
        agreement_px=agreement_px,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        trust=trust,
    )


def _image_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class FaceVerifyResponse(BaseModel):
    is_same_person: bool
    similarity: float
    threshold: float
    margin: float
    confidence: float
    verdict_category: str
    cosine_distance: float
    euclidean_distance: float
    aligned_photo_a: str | None = None
    aligned_photo_b: str | None = None
    model_info: str


class FaceSampleItem(BaseModel):
    id: str
    title: str
    category: str  # "same_person" | "different_people"
    description: str
    image_a_base64: str
    image_b_base64: str


class FaceModelInfoResponse(BaseModel):
    model_name: str
    backbone: str
    projection_head: str
    calibrated_threshold: float
    benchmark_accuracy: str
    dataset_summary: str


@app.post("/api/verify-face", response_model=FaceVerifyResponse)
async def verify_face(
    photo_a: UploadFile = File(...),
    photo_b: UploadFile = File(...),
    threshold: float | None = Form(None),
):
    """Face verification -- evaluates identity similarity between two face photos using
    a frozen DINOv2 backbone + an ArcFace/metric projection head.
    Returns calibrated verdict, confidence, distance metrics, and aligned crops.
    """
    embedder, default_threshold, model_name = _get_face_embedder()
    effective_threshold = threshold if threshold is not None else default_threshold

    try:
        img_a = Image.open(io.BytesIO(await photo_a.read())).convert("RGB")
        img_b = Image.open(io.BytesIO(await photo_b.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded images: {exc}")

    detail = verify_detailed(embedder, img_a, img_b, threshold=effective_threshold)

    aligned_a_b64 = _image_to_base64_jpeg(detail["aligned_a"])
    aligned_b_b64 = _image_to_base64_jpeg(detail["aligned_b"])

    return FaceVerifyResponse(
        is_same_person=detail["is_same_person"],
        similarity=detail["similarity"],
        threshold=detail["threshold"],
        margin=detail["margin"],
        confidence=detail["confidence"],
        verdict_category=detail["verdict_category"],
        cosine_distance=detail["cosine_distance"],
        euclidean_distance=detail["euclidean_distance"],
        aligned_photo_a=aligned_a_b64,
        aligned_photo_b=aligned_b_b64,
        model_info=model_name,
    )


_cached_face_samples: list[FaceSampleItem] | None = None


@app.get("/api/face-samples", response_model=list[FaceSampleItem])
async def get_face_samples():
    """Returns curated verification pairs from LFW and YTF benchmarks for quick demonstration."""
    global _cached_face_samples
    if _cached_face_samples is not None:
        return _cached_face_samples

    repo_root = Path(__file__).resolve().parent.parent
    sample_defs = [
        {
            "id": "jackie_chan_same",
            "title": "Jackie Chan",
            "category": "same_person",
            "description": "LFW benchmark: different poses, lighting, and camera angles.",
            "path_a": repo_root / "datasets/faces/lfw/Jackie_Chan/6378.jpg",
            "path_b": repo_root / "datasets/faces/lfw/Jackie_Chan/2653.jpg",
        },
        {
            "id": "sandra_bullock_same",
            "title": "Sandra Bullock",
            "category": "same_person",
            "description": "LFW benchmark: different facial expressions and focal lengths.",
            "path_a": repo_root / "datasets/faces/lfw/Sandra_Bullock/4292.jpg",
            "path_b": repo_root / "datasets/faces/lfw/Sandra_Bullock/5113.jpg",
        },
        {
            "id": "aaron_eckhart_ytf_same",
            "title": "Aaron Eckhart (Cross-Video)",
            "category": "same_person",
            "description": "YouTube Faces benchmark: extracted from two separate interview videos.",
            "path_a": repo_root / "datasets/faces/ytf_frames/ytf_Aaron_Eckhart/Aaron_Eckhart_0_0.jpg",
            "path_b": repo_root / "datasets/faces/ytf_frames/ytf_Aaron_Eckhart/Aaron_Eckhart_1_0.jpg",
        },
        {
            "id": "stack_vs_bjorn_diff",
            "title": "Robert Stack vs Thomas Bjorn",
            "category": "different_people",
            "description": "LFW benchmark: distinct identities, negative pair.",
            "path_a": repo_root / "datasets/faces/lfw/Robert_Stack/1102.jpg",
            "path_b": repo_root / "datasets/faces/lfw/Thomas_Bjorn/3033.jpg",
        },
        {
            "id": "pfeiffer_vs_glynn_diff",
            "title": "Michelle Pfeiffer vs Kathleen Glynn",
            "category": "different_people",
            "description": "LFW benchmark: distinct identities, negative pair.",
            "path_a": repo_root / "datasets/faces/lfw/Michelle_Pfeiffer/4943.jpg",
            "path_b": repo_root / "datasets/faces/lfw/Kathleen_Glynn/3838.jpg",
        },
    ]

    items: list[FaceSampleItem] = []
    for s in sample_defs:
        if s["path_a"].exists() and s["path_b"].exists():
            try:
                im_a = Image.open(s["path_a"]).convert("RGB")
                im_b = Image.open(s["path_b"]).convert("RGB")
                items.append(
                    FaceSampleItem(
                        id=s["id"],
                        title=s["title"],
                        category=s["category"],
                        description=s["description"],
                        image_a_base64=_image_to_base64_jpeg(im_a),
                        image_b_base64=_image_to_base64_jpeg(im_b),
                    )
                )
            except Exception:
                continue

    _cached_face_samples = items
    return items


@app.get("/api/face-model-info", response_model=FaceModelInfoResponse)
async def get_face_model_info():
    """Returns metadata, architecture specs, and benchmark metrics for the active face model."""
    _, threshold, model_name = _get_face_embedder()
    is_arcface = "ArcFace" in model_name
    return FaceModelInfoResponse(
        model_name=model_name,
        backbone="Meta DINOv2 ViT-B/14 (Frozen, 86M parameters, patch size 14)",
        projection_head="768 -> 256 (ReLU) -> 128 (L2 unit sphere normalization)",
        calibrated_threshold=threshold,
        benchmark_accuracy=(
            "99.40% on 500-pair held-out benchmark (TP: 249/250, TN: 248/250)"
            if is_arcface
            else "89.60% on 500-pair held-out benchmark"
        ),
        dataset_summary=(
            "Trained on combined LFW + YouTube Faces (YTF) corpus: 2,163 identities, 17,385 frames"
            if is_arcface
            else "Trained on LFW corpus (triplet loss)"
        ),
    )


# ── Serve the built React frontend (combined single-container deployment) ──
# In local dev you run the Vite dev server separately and this path won't
# exist, so the mount is skipped and only the API is served on :8000.
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
