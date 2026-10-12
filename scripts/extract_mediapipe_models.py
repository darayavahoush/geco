"""Extract the MediaPipe .tflite models already shipped in frontend/public/mediapipe (no download needed)."""
from pathlib import Path
root = Path(__file__).resolve().parent.parent
data = (root / "frontend/public/mediapipe/face_mesh_solution_packed_assets.data").read_bytes()
out = root / "models/mediapipe"; out.mkdir(parents=True, exist_ok=True)
for name, s, e in [("face_landmark.tflite", 2495952, 3737848)]:   # offsets from face_mesh_solution_packed_assets_loader.js
    (out / name).write_bytes(data[s:e]); print("wrote", out / name, e - s, "bytes")
