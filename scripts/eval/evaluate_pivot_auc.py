"""Like-for-like AUC: geometry score vs deep-embedding baselines on the SAME labelled pairs (pivot csv carries baseline distances).
  PYTHONPATH=src python scripts/eval/evaluate_pivot_auc.py --pairs pivot.csv master.csv --pivot pivot.csv --shapes results/shapes_tta16.pkl"""
import argparse, itertools, collections, pickle
import numpy as np
from geco.face_diff import measure, difference, NoiseModel
from evaluate_face_diff import load_pairs, identities, auc, boot_auc
ap = argparse.ArgumentParser(); ap.add_argument('--pairs', nargs='+', required=True); ap.add_argument('--pivot', required=True); ap.add_argument('--shapes', required=True); a = ap.parse_args()
C = pickle.load(open(a.shapes, 'rb')); allp, _ = load_pairs(a.pairs); pivp, base = load_pairs([a.pivot])
for gate in (False, True):
    keep = {f for f, v in C.items() if (not gate) or (abs(v['pose']['yaw']) <= 15 and abs(v['pose']['pitch']) <= 18)}
    files = sorted(f for f in C if f in keep); ident = identities(files, [p for p in allp if p[0] in keep and p[1] in keep]); M = {f: measure(C[f]['V']) for f in files}
    byid = collections.defaultdict(list)
    for f in files: byid[ident[f]].append(f)
    same = [(x, y) for v in byid.values() for x, y in itertools.combinations(v, 2)]; Ds = np.array([difference(M[x], M[y]) for x, y in same]); sid = np.array([ident[x] for x, _ in same]); cache = {}
    def nm(*ex):
        k = tuple(sorted(set(ex)))
        if k not in cache: cache[k] = NoiseModel.fit(Ds[~np.isin(sid, k)], 't')
        return cache[k]
    P = [(x, y, l) for x, y, l in pivp if x in keep and y in keep]; sc = np.array([np.sqrt((nm(ident[x], ident[y]).z(difference(M[x], M[y])) ** 2).mean()) for x, y, _ in P]); lab = np.array([l for *_, l in P])
    out = {'n': len(P), 'geometry_auc': round(auc(sc[~lab], sc[lab]), 3), 'ci95': [round(v, 3) for v in boot_auc(sc[~lab], sc[lab])]}
    for m in sorted({k for b in base.values() for k in b}):
        v = np.array([base[(x, y)][m] for x, y, _ in P]); out[m] = round(auc(v[~lab], v[lab]), 3)
    print('pose-gated' if gate else 'all poses', out)
