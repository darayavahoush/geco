"""How well do face heads survive glasses / beard changes?  (verification + identification)

Compares one or more projection heads on photos whose appearance differs from the enrolled
photos. Two sources of appearance change:

  --attrs-csv datasets/faces/celeba_attrs.csv   REAL changes (CelebA tags from prepare_celeba.py)
  --root DIR                                    any identity folders (DIR/<name>/*.jpg); glasses and
                                                beard are SYNTHETIC (geco.appearance_aug) -- use this
                                                for LFW/YTF or your own photos

Protocol (identities split into enrolled / unknown so the open-set threshold is honest):
  * gallery   = N 'plain' photos per enrolled identity (optionally + synthetic variants)
  * probes    = every other photo of enrolled identities (grouped by tag: plain/glasses/beard/both)
                + all photos of unknown identities (should be rejected)
  * reports   = verification AUC (plain vs tag, same vs different person), closed-set rank-1,
                and open-set detection-and-identification rate at the threshold giving 1% false accepts.

  python scripts/eval/evaluate_appearance_robustness.py --attrs-csv datasets/faces/celeba_attrs.csv \
      --heads face_head_arcface.pt face_head_robust.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from geco.appearance_aug import random_appearance_change  # noqa: E402
from geco.face import FaceEmbedder, FaceProjectionHead  # noqa: E402
from geco.face_align import extract_face_bbox_and_crop  # noqa: E402
from geco.identify import IMAGE_EXTS  # noqa: E402

TAGS = ["plain", "glasses", "beard", "both"]


def collect_items(a):
    items = []  # dict(path, identity, tag, synth)
    if a.attrs_csv:
        with open(a.attrs_csv) as fh:
            for r in csv.DictReader(fh):
                if r["split"] == a.split and r["tag"] in TAGS:
                    items.append(dict(path=r["path"], identity=r["identity"], tag=r["tag"], synth=None))
    else:
        for d in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
            for f in sorted(x for x in d.iterdir() if x.suffix.lower() in IMAGE_EXTS):
                items.append(dict(path=str(f), identity=d.name, tag="plain", synth=None))
    by_id = defaultdict(list)
    for it in items:
        by_id[it["identity"]].append(it)
    # need >= gallery + 1 photos to be useful
    ids = [i for i, v in by_id.items() if sum(x["tag"] == "plain" for x in v) >= a.gallery_photos and len(v) >= a.gallery_photos + 1]
    random.Random(a.seed).shuffle(ids)
    ids = ids[: a.max_identities]
    return {i: by_id[i] for i in ids}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--attrs-csv"); src.add_argument("--root")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--heads", nargs="+", required=True)
    ap.add_argument("--gallery-photos", type=int, default=2)
    ap.add_argument("--max-identities", type=int, default=300)
    ap.add_argument("--enrolled-fraction", type=float, default=0.7)
    ap.add_argument("--synthetic-gallery", action="store_true", help="also enrol synthetic glasses/beard/both variants of each gallery photo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    people = collect_items(a)
    if len(people) < 4:
        raise SystemExit(f"only {len(people)} usable identities")
    ids = list(people)
    n_enrolled = max(2, int(len(ids) * a.enrolled_fraction))
    enrolled, unknown = ids[:n_enrolled], ids[n_enrolled:]
    print(f"{len(enrolled)} enrolled / {len(unknown)} unknown identities, gallery={a.gallery_photos} photos each")

    # -- one shared frozen-backbone pass (pooled features), heads applied afterwards
    embedder = FaceEmbedder()
    records = []  # dict(identity, tag, role, pooled[768])  role: gallery | probe | unknown | gallery_syn
    def crop_of(path):
        crop, meta = extract_face_bbox_and_crop(Image.open(path).convert("RGB"))
        return crop if meta.get("method") == "yunet_dnn" else None

    for k, ident in enumerate(ids):
        photos = people[ident]
        plain = [p for p in photos if p["tag"] == "plain"]
        gal = plain[: a.gallery_photos] if ident in enrolled else []
        gal_paths = {p["path"] for p in gal}
        for p in photos:
            crop = crop_of(p["path"])
            if crop is None:
                continue
            pooled = embedder.pooled_from_aligned(crop).cpu()
            if p["path"] in gal_paths:
                records.append(dict(identity=ident, tag="plain", role="gallery", pooled=pooled))
                if a.synthetic_gallery:
                    for m in ("glasses", "beard", "both"):
                        records.append(dict(identity=ident, tag=m, role="gallery_syn",
                                            pooled=embedder.pooled_from_aligned(random_appearance_change(crop, rng, mode=m)[0]).cpu()))
            else:
                role = "probe" if ident in enrolled else "unknown"
                records.append(dict(identity=ident, tag=p["tag"], role=role, pooled=pooled))
                if a.root:  # synthetic probes from clean photos
                    for m in ("glasses", "beard", "both"):
                        records.append(dict(identity=ident, tag=m, role=role,
                                            pooled=embedder.pooled_from_aligned(random_appearance_change(crop, rng, mode=m)[0]).cpu()))
        if (k + 1) % 25 == 0:
            print(f"  embedded {k + 1}/{len(ids)} identities", flush=True)

    pooled_all = torch.stack([r["pooled"] for r in records])
    ident_arr = np.array([r["identity"] for r in records])
    tag_arr = np.array([r["tag"] for r in records])
    role_arr = np.array([r["role"] for r in records])
    results = {}

    for head_path in a.heads:
        head = FaceProjectionHead()
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
        head.eval()
        with torch.no_grad():
            E = head(pooled_all.to(embedder.device)).cpu().numpy()
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        res = {}

        # verification AUC: plain gallery photo vs (probe of tag t), same vs different identity
        g_idx = np.where(role_arr == "gallery")[0]
        for t in TAGS[1:] if a.attrs_csv else TAGS:
            p_idx = np.where((role_arr == "probe") & (tag_arr == t))[0]
            if len(p_idx) == 0:
                continue
            scores, labels = [], []
            for pi in p_idx:
                same = g_idx[ident_arr[g_idx] == ident_arr[pi]]
                diff = g_idx[ident_arr[g_idx] != ident_arr[pi]]
                if len(same) == 0 or len(diff) == 0:
                    continue
                scores.append(float(E[same[0]] @ E[pi])); labels.append(1)
                scores.append(float(E[rng.choice(diff)] @ E[pi])); labels.append(0)
            if len(set(labels)) == 2:
                res[f"verify_auc_{t}"] = float(roc_auc_score(labels, scores))

        # identification: best template per person
        templ_idx = np.where((role_arr == "gallery") | (role_arr == "gallery_syn"))[0]
        templ_people = ident_arr[templ_idx]
        def best_person(i):
            sims = E[templ_idx] @ E[i]
            best = {}
            for s, p in zip(sims, templ_people):
                if p not in best or s > best[p]:
                    best[p] = s
            top = max(best, key=best.get)
            return top, float(best[top])

        unk = np.where(role_arr == "unknown")[0]
        unk_scores = np.array([best_person(i)[1] for i in unk]) if len(unk) else np.array([])
        thr_1pct = float(np.quantile(unk_scores, 0.99)) if len(unk_scores) else 0.95
        res["threshold_at_1pct_far"] = thr_1pct
        for t in TAGS:
            p_idx = np.where((role_arr == "probe") & (tag_arr == t))[0]
            if len(p_idx) == 0:
                continue
            out = [best_person(i) for i in p_idx]
            correct = np.array([o[0] == ident_arr[i] for o, i in zip(out, p_idx)])
            sc = np.array([o[1] for o in out])
            res[f"rank1_{t}"] = float(correct.mean())
            res[f"dir_at_1pct_far_{t}"] = float((correct & (sc >= thr_1pct)).mean())
            res[f"n_{t}"] = int(len(p_idx))
        results[head_path] = res

    keys = [f"{k}_{t}" for k in ("verify_auc", "rank1", "dir_at_1pct_far") for t in TAGS]
    print("\n" + "=" * 100)
    print(f"{'metric':<26}" + "".join(f"{Path(h).name:>26}" for h in a.heads))
    print("-" * 100)
    for k in keys:
        if any(k in results[h] for h in a.heads):
            print(f"{k:<26}" + "".join(f"{results[h].get(k, float('nan')):>26.4f}" for h in a.heads))
    print(f"{'threshold @1% FAR':<26}" + "".join(f"{results[h]['threshold_at_1pct_far']:>26.4f}" for h in a.heads))
    print("=" * 100)
    print("rank1 = closed-set top-1 | dir_at_1pct_far = correct AND accepted at the threshold that lets 1% of unknown people in")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
