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

import io
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geco.features import get_device, load_dinov2  # noqa: E402
from geco.keypoints import transfer_keypoint  # noqa: E402
from geco.matching import MatchResult, geco_match  # noqa: E402
from geco.visualize import render_confidence_heatmap, render_match_heatmap, render_pca_map  # noqa: E402

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
