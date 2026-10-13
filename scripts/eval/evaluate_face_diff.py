"""Evaluate feature-level 3D face difference attribution (reliability, calibration, discrimination).

Data format = deepface unit-test dataset (images + CSV of labelled pairs). Any dataset with
``file_x,file_y,decision(Yes/No)`` CSVs works.

  PYTHONPATH=src python scripts/eval/evaluate_face_diff.py --images DIR --pairs pivot.csv [master.csv ...] --out results/
"""
from __future__ import annotations
import argparse, csv, collections, itertools, json, pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image
from geco.face_mp3d import reconstruct
from geco.face_diff import FEATURES, NAMES, measure, difference, robust_sigma, bh, NoiseModel


def load_pairs(paths):
    pairs, base = [], {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            fx, fy = Path(r['file_x']).name, Path(r['file_y']).name
            dec = r.get('decision', r.get('Decision')) == 'Yes'
            pairs.append((fx, fy, dec)); base[(fx, fy)] = {k: float(v) for k, v in r.items() if k.endswith('_cosine')}
    return pairs, base


def identities(files, pairs):
    par = {f: f for f in files}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b, y in pairs:
        if y: par[find(a)] = find(b)
    bad = [(a, b) for a, b, y in pairs if (not y) and find(a) == find(b)]
    assert not bad, f'label conflicts: {bad[:3]}'
    return {f: find(f) for f in files}


def z2p(z):
    from scipy import stats
    return 2 * stats.norm.sf(np.abs(z))


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 ties  (pos = different-person pairs)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def boot_auc(pos, neg, n=300, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        out.append(auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
    return np.percentile(out, [2.5, 97.5]).tolist()


def icc1(X, groups):
    """One-way random ICC(1,1) with unbalanced groups. X [N], groups [N]."""
    g = collections.defaultdict(list)
    for x, k in zip(X, groups): g[k].append(x)
    g = {k: np.array(v) for k, v in g.items() if len(v) >= 2}
    N = sum(len(v) for v in g.values()); k = len(g); gm = np.concatenate(list(g.values())).mean()
    ssb = sum(len(v) * (v.mean() - gm) ** 2 for v in g.values()); ssw = sum(((v - v.mean()) ** 2).sum() for v in g.values())
    msb, msw = ssb / (k - 1), ssw / (N - k); n0 = (N - sum(len(v) ** 2 for v in g.values()) / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--pairs', nargs='+', required=True)
    ap.add_argument('--out', default='results'); ap.add_argument('--cache', default=None); ap.add_argument('--tta', type=int, default=16)
    ap.add_argument('--noise', choices=['gauss', 't'], default='t'); ap.add_argument('--max-yaw', type=float, default=None); ap.add_argument('--max-pitch', type=float, default=None); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs, base = load_pairs(a.pairs); files = sorted({f for p in pairs for f in p[:2]})
    cache = Path(a.cache) if a.cache else out / f'shapes_tta{a.tta}.pkl'
    if cache.exists(): C = pickle.load(open(cache, 'rb'))
    else:
        C = {}
        for f in files:
            r = reconstruct(Image.open(Path(a.images) / f).convert('RGB'), tta=a.tta); C[f] = dict(V=r['V'][:468].astype(np.float32), pose=r['pose'])
        pickle.dump(C, open(cache, 'wb'))
    S = {f: v['V'] for f, v in C.items()}
    if a.max_yaw is not None or a.max_pitch is not None:      # quality gate: drop extreme-pose images
        keep = {f for f, v in C.items() if abs(v['pose']['yaw']) <= (a.max_yaw or 1e9) and abs(v['pose']['pitch']) <= (a.max_pitch or 1e9)}
        files = [f for f in files if f in keep]; pairs = [p for p in pairs if p[0] in keep and p[1] in keep]; print(f'gate kept {len(files)} images')
    ident = identities(files, pairs); M = {f: measure(S[f]) for f in files}
    byid = collections.defaultdict(list)
    for f in files: byid[ident[f]].append(f)
    print(f'{len(files)} images, {len(byid)} identities, {len(pairs)} labelled pairs')

    # ---- all within-/between-identity pairs (exhaustive, for calibration + extra power)
    same = [(x, y, ident[x]) for k, v in byid.items() for x, y in itertools.combinations(v, 2)]
    diff = [(x, y) for x, y in itertools.combinations(files, 2) if ident[x] != ident[y]]
    Dsame = np.array([difference(M[x], M[y]) for x, y, _ in same]); same_id = np.array([k for *_, k in same])
    Dsame_signed_sym = np.vstack([Dsame, -Dsame])            # symmetrise: pair order is arbitrary
    cache_sigma = {}
    def noise_excl(*excl):                                      # leave-identity-out within-person noise model
        key = tuple(sorted(set(excl)))
        if key not in cache_sigma: cache_sigma[key] = NoiseModel.fit(Dsame[~np.isin(same_id, key)], a.noise)
        return cache_sigma[key]
    def pvec(x, y): return noise_excl(ident[x], ident[y]).pvalues(difference(M[x], M[y]))
    def zvec(x, y): return noise_excl(ident[x], ident[y]).z(difference(M[x], M[y]))

    res = {'n_images': len(files), 'n_identities': len(byid), 'n_same_pairs_exhaustive': len(same), 'n_diff_pairs_exhaustive': len(diff)}

    # ---- E1 reliability: ICC(1,1) with identity-cluster bootstrap CI
    rng = np.random.default_rng(0); ids = [k for k, v in byid.items() if len(v) >= 2]; icc = {}
    for k, f in enumerate(FEATURES):
        X = [v for i in ids for v in [(np.log(M[x][k]) if f.kind == 'ratio' else M[x][k]) for x in byid[i]]]; G = [i for i in ids for _ in byid[i]]
        pt = icc1(X, G); bs = []
        for _ in range(300):
            pick = rng.choice(len(ids), len(ids)); XX, GG = [], []
            for j, p in enumerate(pick):
                for x in byid[ids[p]]: XX.append(np.log(M[x][k]) if f.kind == 'ratio' else M[x][k]); GG.append(j)
            bs.append(icc1(XX, GG))
        icc[f.name] = dict(icc=pt, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    res['icc'] = icc

    # ---- E2 calibration of the significance test (held-out identities)
    Zs = np.array([zvec(x, y) for x, y, _ in same]); Zd = np.array([zvec(x, y) for x, y in diff])
    res['calibration'] = dict(
        same_person_fpr_abs_z_gt_1_96=float((np.abs(Zs) > 1.96).mean()), same_person_fpr_abs_z_gt_2_58=float((np.abs(Zs) > 2.58).mean()),
        nominal_1_96=0.05, nominal_2_58=0.01,
        same_pairs_with_any_bh_flag=float(np.mean([bh(z2p(z)).any() for z in Zs])),
        diff_pairs_with_any_bh_flag=float(np.mean([bh(z2p(z)).any() for z in Zd])),
        mean_flags_same=float(np.mean([bh(z2p(z)).sum() for z in Zs])),
        mean_flags_diff=float(np.mean([bh(z2p(z)).sum() for z in Zd])))
    res['per_feature_flag_rate'] = {n: dict(same=float((np.abs(Zs[:, k]) > 2.58).mean()), diff=float((np.abs(Zd[:, k]) > 2.58).mean())) for k, n in enumerate(NAMES)}
    nm = NoiseModel.fit(Dsame, a.noise); thr = nm.threshold(0.01)
    res['min_detectable_difference'] = {n: (f'{100 * (np.exp(thr[k]) - 1):.1f}%' if f.kind == 'ratio' else f'{100 * thr[k]:.1f}% of face width' if f.kind == 'lin' else f'{thr[k]:.1f} deg') for k, (n, f) in enumerate(zip(NAMES, FEATURES))}
    res['noise_model'] = {n: dict(df=float(nm.df[k]), scale=float(nm.scale[k])) for k, n in enumerate(NAMES)}; res['config'] = dict(tta=a.tta, noise=a.noise)

    # ---- E3 discrimination: AUC of geometric scores vs deep-embedding baselines on the labelled benchmark pairs
    def scores(plist):
        Z = np.array([zvec(x, y) for x, y, *_ in plist]); return np.sqrt((Z ** 2).mean(1)), Z
    def maha(plist):
        out = []
        for x, y, *_ in plist:
            d = Dsame[~np.isin(same_id, [ident[x], ident[y]])]; d = np.vstack([d, -d]); C = np.cov(d.T); C = 0.8 * C + 0.2 * np.diag(np.diag(C))
            v = difference(M[x], M[y]); out.append(float(np.sqrt(v @ np.linalg.solve(C, v) / len(v))))
        return np.array(out)
    lab = np.array([y for *_, y in pairs]); rms, Z = scores(pairs); mh = maha(pairs)
    pos, neg = rms[~lab], rms[lab]
    disc = {'geometry_rms_z': dict(auc=auc(pos, neg), ci=boot_auc(pos, neg)), 'geometry_mahalanobis': dict(auc=auc(mh[~lab], mh[lab]), ci=boot_auc(mh[~lab], mh[lab]))}
    for k, n in enumerate(NAMES): disc[f'feature:{n}'] = dict(auc=auc(np.abs(Z[~lab, k]), np.abs(Z[lab, k])))
    for m in sorted({kk for b in base.values() for kk in b}):
        v = np.array([base[(x, y)].get(m, np.nan) for x, y, _ in pairs]); ok = ~np.isnan(v)
        if ok.sum() > 0: disc[f'baseline:{m}'] = dict(auc=auc(v[ok & ~lab], v[ok & lab]), ci=boot_auc(v[ok & ~lab], v[ok & lab]), n=int(ok.sum()))
    ex_rms = np.array([np.sqrt((z ** 2).mean()) for z in np.vstack([Zd, Zs])]); ex_lab = np.r_[np.zeros(len(Zd), bool), np.ones(len(Zs), bool)]
    disc['geometry_rms_z_exhaustive'] = dict(auc=auc(ex_rms[~ex_lab], ex_rms[ex_lab]), ci=boot_auc(ex_rms[~ex_lab], ex_rms[ex_lab]), n_same=len(Zs), n_diff=len(Zd))
    res['discrimination'] = disc; res['n_benchmark_pairs'] = dict(same=int(lab.sum()), different=int((~lab).sum()))
    tag = f'tta{a.tta}_{a.noise}' + (f'_gate{int(a.max_yaw or 0)}_{int(a.max_pitch or 0)}' if (a.max_yaw or a.max_pitch) else '')
    res['n_benchmark_images_after_gate'] = len(files); json.dump(res, open(out / f'results_stats_{tag}.json', 'w'), indent=1); print(json.dumps({k: res[k] for k in ('calibration',)}, indent=1))
    print('AUC:', {k: round(v['auc'], 3) for k, v in disc.items() if not k.startswith('feature:')})
    print('ICC:', {k: round(v['icc'], 2) for k, v in icc.items()})


if __name__ == '__main__':
    main()
