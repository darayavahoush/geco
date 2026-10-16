"""Lay CelebA out as identity folders + an appearance-tag CSV, for glasses/beard robustness.

CelebA ships identity labels (identity_CelebA.txt) and 40 binary attributes
(list_attr_celeba.txt) including Eyeglasses, No_Beard, Goatee, Mustache and
5_o_Clock_Shadow -- exactly what is needed to find pairs of the SAME person
with and without glasses / a beard.

Input  (--celeba-root): a folder containing
    img_align_celeba/*.jpg   identity_CelebA.txt   list_attr_celeba.txt
Output (--out-root):
    celeba_train/<identity>/<file>.jpg     identities used for training
    celeba_eval/<identity>/<file>.jpg      held-out identities (never trained on)
    celeba_attrs.csv                        path,identity,split,tag   (tag: plain|glasses|beard|both|other)

CelebA is licensed for non-commercial research use only.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def read_attrs(path: Path) -> dict[str, dict[str, int]]:
    lines = path.read_text().splitlines()
    names = lines[1].split()
    out = {}
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) == len(names) + 1:
            out[parts[0]] = dict(zip(names, (int(v) for v in parts[1:])))
    return out


def appearance_tag(a: dict[str, int]) -> str:
    glasses = a["Eyeglasses"] == 1
    beard = a["No_Beard"] == -1 and (a["Goatee"] == 1 or a["Mustache"] == 1 or a["5_o_Clock_Shadow"] == 1)
    clean_face = a["No_Beard"] == 1
    if glasses and beard:
        return "both"
    if glasses and clean_face:
        return "glasses"
    if beard and not glasses:
        return "beard"
    if clean_face and not glasses:
        return "plain"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--min-images", type=int, default=8, help="drop identities with fewer images")
    ap.add_argument("--max-identities", type=int, default=2000, help="cap train+eval identities (CPU-friendly default)")
    ap.add_argument("--eval-identities", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    img_dir = a.celeba_root / "img_align_celeba"
    ident_path = a.celeba_root / "identity_CelebA.txt"
    attr_path = a.celeba_root / "list_attr_celeba.txt"
    for p in (img_dir, ident_path, attr_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    attrs = read_attrs(attr_path)
    by_id: dict[str, list[str]] = defaultdict(list)
    for ln in ident_path.read_text().splitlines():
        f, i = ln.split()
        if f in attrs:
            by_id[i].append(f)
    ids = sorted(i for i, fs in by_id.items() if len(fs) >= a.min_images)
    random.Random(a.seed).shuffle(ids)
    ids = ids[: a.max_identities]
    eval_ids, train_ids = set(ids[: a.eval_identities]), set(ids[a.eval_identities:])
    print(f"{len(ids)} identities ({len(train_ids)} train / {len(eval_ids)} eval)")

    rows = []
    for split, group in (("train", train_ids), ("eval", eval_ids)):
        for i in sorted(group):
            d = a.out_root / f"celeba_{split}" / f"celeba_{i}"
            d.mkdir(parents=True, exist_ok=True)
            for f in by_id[i]:
                dst = d / f
                if not dst.exists():
                    shutil.copy(img_dir / f, dst)
                rows.append((str(dst), f"celeba_{i}", split, appearance_tag(attrs[f])))

    csv_path = a.out_root / "celeba_attrs.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "identity", "split", "tag"])
        w.writerows(rows)

    counts = defaultdict(int)
    for _, _, split, tag in rows:
        counts[(split, tag)] += 1
    for split in ("train", "eval"):
        print(split, {t: counts[(split, t)] for t in ("plain", "glasses", "beard", "both", "other")})
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
