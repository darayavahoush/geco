"""Closed-loop accuracy test for the 3D face: render at novel yaws, re-detect 468 landmarks on the render,
compare with the mesh's own projection. Usage: python scripts/eval/evaluate_face3d.py photo.jpg"""
import sys, numpy as np
from PIL import Image
from geco.face_mp3d import reconstruct, residual_map, export_obj
img = Image.open(sys.argv[1]); r = reconstruct(img)
print("pose of input (deg):", {k: round(v, 1) for k, v in r["pose"].items()})
for yaw in (0, 15, 30, 45):
    e = residual_map(r, yaw)
    print(f"yaw {yaw:>3}: NME {np.nanmean(e):.2f}% IOD  (median {np.nanmedian(e):.2f}, p95 {np.nanpercentile(e,95):.2f})" if not np.isnan(e).all() else f"yaw {yaw:>3}: face not re-detected")
export_obj(r, "face3d.obj"); print("wrote face3d.obj")
