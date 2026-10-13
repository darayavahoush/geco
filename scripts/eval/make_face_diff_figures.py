"""Figures for docs/FACE_DIFFERENCE_PAPER.md. Run after evaluate_face_diff.py and evaluate_face_edits.py.

  PYTHONPATH=src python scripts/eval/make_face_diff_figures.py --results results/ --pairs pivot.csv master.csv --out docs/figures
"""
from __future__ import annotations
import argparse, collections, itertools, json, pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.cm as cm
from geco.face_mp3d import render, TRI
from geco.face_diff import (FEATURES, NAMES, measure, difference, NoiseModel, compare, explain, procrustes_displacement, RELIABILITY_ICC, MIN_ICC)
from evaluate_face_diff import load_pairs, identities

NICE = lambda n: n.replace('_', ' ')


def fig_reliability(res, out):
    icc = res['icc']; mdd = res['min_detectable_difference']; names = sorted(NAMES, key=lambda n: -icc[n]['icc'])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    y = np.arange(len(names)); v = np.array([icc[n]['icc'] for n in names]); lo = np.array([icc[n]['lo'] for n in names]); hi = np.array([icc[n]['hi'] for n in names])
    ax[0].barh(y, v, color=['#2a9d8f' if x >= 0.5 else '#e9c46a' if x >= MIN_ICC else '#c9c9c9' for x in v]); ax[0].errorbar(v, y, xerr=[v - lo, hi - v], fmt='none', ecolor='k', lw=1, capsize=2)
    ax[0].axvline(MIN_ICC, ls='--', c='gray', lw=1); ax[0].set_yticks(y); ax[0].set_yticklabels([NICE(n) for n in names]); ax[0].invert_yaxis(); ax[0].set_xlabel('ICC(1,1) across photos of one person (95% CI)')
    ax[0].set_title('(a) Which features are stable enough to compare?', fontsize=10, loc='left')
    ax[1].barh(y, [float(str(mdd[n]).split()[0].rstrip('%')) if 'deg' not in mdd[n] else float(str(mdd[n]).split()[0]) for n in names], color='#8d99ae')
    for i, n in enumerate(names): ax[1].text(float(str(mdd[n]).split()[0].rstrip('%')) + 0.5, i, mdd[n], va='center', fontsize=7)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].set_xlabel('min. detectable difference (2-sided 1%; unit as labelled)'); ax[1].set_xlim(0, 135)
    ax[1].set_title('(b) How large must a difference be to exceed photo-to-photo variation?'.replace('How large must a difference be to exceed','Size needed to exceed'), fontsize=10, loc='left')
    plt.tight_layout(); plt.savefig(out / 'fig1_reliability_detectability.png', dpi=170); plt.close()


def fig_edits(rows, out):
    E = [r for r in rows if not r['edit'].startswith('CONTROL')]; names = list(dict.fromkeys(r['edit'] for r in E)); mags = sorted({r['mag'] for r in E})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4)); w = 0.8 / len(mags)
    for j, m in enumerate(mags):
        ax[0].bar(np.arange(len(names)) + j * w, [np.mean([r['dir_ok'] for r in E if r['edit'] == n and r['mag'] == m]) for n in names], w, label=f'dose {m:.2f}')
        ax[1].bar(np.arange(len(names)) + j * w, [np.mean([r['realized'] * r['sign'] for r in E if r['edit'] == n and r['mag'] == m]) / m for n in names], w)
        ax[2].bar(np.arange(len(names)) + j * w, [np.mean([r['detected_inst'] for r in E if r['edit'] == n and r['mag'] == m]) for n in names], w)
    for a, t in zip(ax, ['(a) direction of measured change is correct', '(b) realised / applied change (landmark-network "gain")', '(c) edit flagged under instrument-noise null']):
        a.set_xticks(np.arange(len(names)) + w); a.set_xticklabels([NICE(n) for n in names], rotation=40, ha='right', fontsize=8); a.set_title(t, fontsize=9, loc='left')
    ax[0].set_ylim(0, 1.05); ax[0].axhline(0.5, c='gray', ls=':'); ax[0].legend(fontsize=7); ax[2].set_ylim(0, 1.05); plt.tight_layout(); plt.savefig(out / 'fig2_edit_recovery.png', dpi=170); plt.close()


def fig_pairs(a, out):
    pairs, _ = load_pairs(a.pairs); C = pickle.load(open(a.shapes, 'rb'))
    keep = {f for f, v in C.items() if abs(v['pose']['yaw']) <= 15 and abs(v['pose']['pitch']) <= 18}
    files = sorted(f for f in {p for pr in pairs for p in pr[:2]} if f in keep); pairs = [p for p in pairs if p[0] in keep and p[1] in keep]; ident = identities(files, pairs)
    M = {f: measure(C[f]['V']) for f in files}; byid = collections.defaultdict(list)
    for f in files: byid[ident[f]].append(f)
    same = [(x, y) for v in byid.values() for x, y in itertools.combinations(v, 2)]; D = np.array([difference(M[x], M[y]) for x, y in same]); sid = np.array([ident[x] for x, _ in same])
    _c = {}
    def nm(*ex):
        k = tuple(sorted(set(ex)))
        if k not in _c: _c[k] = NoiseModel.fit(D[~np.isin(sid, k)], 't')
        return _c[k]
    cand_d, cand_s = [], []
    for x, y in itertools.combinations(files, 2):
        n_ = nm(ident[x], ident[y]); r = compare(C[x]['V'], C[y]['V'], n_); k = sum(q.evidence == 'significant' for q in r)
        (cand_s if ident[x] == ident[y] else cand_d).append((k, max(abs(q.z) for q in r if q.icc >= MIN_ICC), x, y))
    dsel = sorted(cand_d, reverse=True)[0]; ssel = sorted(cand_s, key=lambda t: (t[0], t[1]))[0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    tri = TRI[(TRI < 468).all(1)]
    for row, (tag, sel) in enumerate([('DIFFERENT people', dsel), ('SAME person (two photos)', ssel)]):
        _, _, x, y = sel; n_ = nm(ident[x], ident[y]); r = compare(C[x]['V'], C[y]['V'], n_, label_a='A', label_b='B')
        disp, rms = procrustes_displacement(C[x]['V'], C[y]['V']); mag = 100 * np.linalg.norm(disp, axis=1); vmax = 6.0
        col = (cm.turbo(np.clip(mag / vmax, 0, 1))[:, :3] * 255).astype(np.float32); gray = np.full((468, 3), 150, np.float32)
        for c, (Vv, vc, ttl) in enumerate([(C[x]['V'], gray, 'A  (geometry only)'), (C[x]['V'], col, 'A, coloured by displacement to B')]):
            im, _, _ = render(Vv.astype(np.float64), np.zeros((468, 2), np.float32), np.zeros((4, 4, 3), np.uint8), (0, 0, 0), 0, 0, size=420, vert_color=vc, tri=tri)
            axs[row, c].imshow(im); axs[row, c].axis('off'); axs[row, c].set_title(ttl, fontsize=9)
        axs[row, 2].axis('off'); txt = explain(r)
        axs[row, 2].text(0, 1, f'{tag}\nProcrustes shape distance = {100 * rms:.2f}% of centroid size\n\n' + txt, va='top', fontsize=7.6, family='monospace', wrap=True)
    sm = cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(0, vmax)); fig.colorbar(sm, ax=axs[:, 1], fraction=0.02, label='per-vertex displacement after similarity alignment (% of centroid size)')
    plt.savefig(out / 'fig3_pair_explanations.png', dpi=170, bbox_inches='tight'); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', required=True); ap.add_argument('--pairs', nargs='+', required=True); ap.add_argument('--shapes', required=True)
    ap.add_argument('--stats', default='results_stats_tta16_t_gate15_18.json'); ap.add_argument('--out', default='docs/figures'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); R = Path(a.results)
    fig_reliability(json.load(open(R / a.stats)), out); fig_edits(json.load(open(R / 'results_edits.json')), out); fig_pairs(a, out); print('figures ->', out)


if __name__ == '__main__':
    main()
