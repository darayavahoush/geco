#!/usr/bin/env python
"""Command-line face identification: enroll people, then ask "who is this?".

  # enroll one person from a few photos (add --synthetic 3 to also store glasses/beard variants)
  python scripts/identify.py enroll --gallery gallery.npz --name Alice a1.jpg a2.jpg a3.jpg

  # enroll a whole folder laid out as people/<name>/*.jpg
  python scripts/identify.py enroll-dir --gallery gallery.npz --root people/ --synthetic 3

  # who is in these photos?
  python scripts/identify.py identify --gallery gallery.npz probe1.jpg probe2.jpg

  # other helpers
  python scripts/identify.py list --gallery gallery.npz
  python scripts/identify.py remove --gallery gallery.npz --name Alice
  python scripts/identify.py compare a.jpg b.jpg            # 1:1 similarity

Use --head face_head_robust.pt (from train_face_arcface_robust.py) for the glasses/beard-robust head.
Only enroll people who have agreed to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from geco.identify import (  # noqa: E402
    DEFAULT_MARGIN, DEFAULT_THRESHOLD, Gallery, NoFaceFound, enroll_photos, list_images, load_embedder, make_embed_fn,
)


def cmd_enroll(a):
    g = Gallery.load(a.gallery)
    photos = [p for x in a.photos for p in list_images(x)]
    if not photos:
        raise SystemExit("no images found")
    added, skipped = enroll_photos(g, load_embedder(a.head), a.name, photos, synthetic_per_photo=a.synthetic)
    g.save(a.gallery)
    print(f"{a.name}: +{added} templates (now {g.people().get(a.name, 0)}); gallery has {len(g.people())} people -> {a.gallery}")
    for s in skipped:
        print("  skipped", s)


def cmd_enroll_dir(a):
    g = Gallery.load(a.gallery)
    embedder = load_embedder(a.head)
    root = Path(a.root)
    people = sorted(d for d in root.iterdir() if d.is_dir())
    if not people:
        raise SystemExit(f"no person folders under {root} (expected {root}/<name>/*.jpg)")
    for d in people:
        photos = list_images(d)
        added, skipped = enroll_photos(g, embedder, d.name, photos, synthetic_per_photo=a.synthetic)
        print(f"{d.name}: {len(photos)} photos -> +{added} templates" + (f", {len(skipped)} skipped" if skipped else ""))
        for s in skipped:
            print("   skipped", s)
    g.save(a.gallery)
    print(f"saved {len(g.people())} people -> {a.gallery}")


def cmd_identify(a):
    g = Gallery.load(a.gallery)
    if not len(g):
        raise SystemExit(f"gallery {a.gallery} is empty or missing -- enroll first")
    embed = make_embed_fn(load_embedder(a.head))
    out = []
    for path in [p for x in a.probes for p in list_images(x)]:
        try:
            emb, meta = embed(Image.open(path).convert("RGB"))
        except NoFaceFound as exc:
            print(f"{path}: NO FACE ({exc})")
            out.append({"photo": str(path), "verdict": "no_face"})
            continue
        r = g.identify(emb, threshold=a.threshold, margin=a.margin, top_k=a.top_k)
        who = {"match": r.name, "ambiguous": f"ambiguous (closest: {r.name})", "unknown": "UNKNOWN"}[r.verdict]
        print(f"{path}: {who}   score={r.score:.4f}  margin={r.margin:.4f}  det={meta['confidence']}%")
        for m in r.ranking:
            print(f"    {m.name:<24} {m.score:.4f}   best template: {m.best_source}")
        out.append({"photo": str(path), "verdict": r.verdict, "name": r.name, "score": r.score,
                    "ranking": [m.__dict__ for m in r.ranking]})
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print("wrote", a.json)


def cmd_list(a):
    g = Gallery.load(a.gallery)
    if not len(g):
        print("(empty)")
    for name, n in sorted(g.people().items()):
        print(f"{name:<28} {n} templates")


def cmd_remove(a):
    g = Gallery.load(a.gallery)
    n = g.remove(a.name)
    g.save(a.gallery)
    print(f"removed {n} templates for {a.name}")


def cmd_compare(a):
    embed = make_embed_fn(load_embedder(a.head))
    ea, _ = embed(Image.open(a.a).convert("RGB"))
    eb, _ = embed(Image.open(a.b).convert("RGB"))
    sim = float(np.dot(ea, eb))
    print(f"cosine similarity = {sim:.4f}  ->  {'SAME person' if sim >= a.threshold else 'DIFFERENT people'} (threshold {a.threshold})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", default=None, help="projection-head weights (default: face_head_arcface.pt)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("enroll"); p.add_argument("--gallery", required=True); p.add_argument("--name", required=True)
    p.add_argument("--synthetic", type=int, default=0, help="also store N synthetic appearance variants per photo (1=glasses, 2=+beard, 3=+both)")
    p.add_argument("photos", nargs="+"); p.set_defaults(fn=cmd_enroll)

    p = sub.add_parser("enroll-dir"); p.add_argument("--gallery", required=True); p.add_argument("--root", required=True)
    p.add_argument("--synthetic", type=int, default=0); p.set_defaults(fn=cmd_enroll_dir)

    p = sub.add_parser("identify"); p.add_argument("--gallery", required=True)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD); p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--top-k", type=int, default=3); p.add_argument("--json", default=None); p.add_argument("probes", nargs="+")
    p.set_defaults(fn=cmd_identify)

    p = sub.add_parser("list"); p.add_argument("--gallery", required=True); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("remove"); p.add_argument("--gallery", required=True); p.add_argument("--name", required=True); p.set_defaults(fn=cmd_remove)
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
