"""Controlled-edit recovery: apply a KNOWN local geometric edit to a photo, then ask the difference method
which feature changed and in which direction. Ground truth = the edit we applied.

  PYTHONPATH=src python scripts/eval/evaluate_face_edits.py --images DIR --pairs pivot.csv master.csv --out results/

Edits are image-space warps (Gaussian-weighted local scaling/shift). They test the measurement + attribution pipeline
(landmark network -> pose normalisation -> features -> calibrated significance), not 3D ground-truth accuracy.
"""
from __future__ import annotations
import argparse, collections, itertools, json, pickle
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from geco.face_mp3d import reconstruct
from geco.face_diff import FEATURES, NAMES, measure, difference, bh, NoiseModel
from evaluate_face_diff import load_pairs, identities

IDX = dict(oR=33, iR=133, oL=263, iL=362, uR=159, lR=145, uL=386, lL=374, bR=105, bL=334, alR=129, alL=358, mR=61, mL=291,
           lu_b=13, ll_b=17, chin=152, top=10, zR=234, zL=454, gR=172, gL=397)


def _warp(img: np.ndarray, ux, uy):
    """Backward warp with forward displacement field (ux,uy) given on the pixel grid (first-order inverse)."""
    h, w = img.shape[:2]; gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx - ux.astype(np.float32), gy - uy.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _g(gx, gy, cx, cy, sx, sy):
    return np.exp(-0.5 * (((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2))


# name -> (target feature, features structurally coupled to the target (lenient set), function(img, P, m) -> edited img)
def make_edits():
    def grid(img): h, w = img.shape[:2]; return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    def p(P, k): return P[IDX[k], :2]

    def nose_width(img, P, m):
        gx, gy = grid(img); c = (p(P, 'alR') + p(P, 'alL')) / 2; wd = np.linalg.norm(p(P, 'alR') - p(P, 'alL'))
        return _warp(img, m * (gx - c[0]) * _g(gx, gy, c[0], c[1], 0.9 * wd, 0.9 * wd), 0 * gx)

    def mouth_width(img, P, m):
        gx, gy = grid(img); c = (p(P, 'mR') + p(P, 'mL')) / 2; wd = np.linalg.norm(p(P, 'mR') - p(P, 'mL'))
        return _warp(img, m * (gx - c[0]) * _g(gx, gy, c[0], c[1], 0.8 * wd, 0.5 * wd), 0 * gx)

    def eye_opening(img, P, m):
        gx, gy = grid(img); uy = 0 * gx
        for s in 'RL':
            c = (p(P, 'o' + s) + p(P, 'i' + s)) / 2; wd = np.linalg.norm(p(P, 'o' + s) - p(P, 'i' + s))
            uy = uy + m * (gy - c[1]) * _g(gx, gy, c[0], c[1], 0.7 * wd, 0.45 * wd)
        return _warp(img, 0 * gx, uy)

    def jaw_width(img, P, m):
        gx, gy = grid(img); c = (p(P, 'gR') + p(P, 'gL')) / 2; fw = abs(p(P, 'zR')[0] - p(P, 'zL')[0])
        return _warp(img, m * (gx - c[0]) * _g(gx, gy, c[0], c[1], 0.7 * fw, 0.18 * fw), 0 * gx)

    def chin_height(img, P, m):
        gx, gy = grid(img); y0 = p(P, 'll_b')[1]; cx = p(P, 'chin')[0]; fh = abs(p(P, 'top')[1] - p(P, 'chin')[1])
        ramp = 1.0 / (1.0 + np.exp(-(gy - y0) / (0.03 * fh)))
        return _warp(img, 0 * gx, m * (gy - y0) * ramp * _g(gx, gy, cx, 0 * gy, 0.45 * fh, 1e9))

    def brow_height(img, P, m):
        gx, gy = grid(img); uy = 0 * gx
        for s in 'RL':
            b, e = p(P, 'b' + s), (p(P, 'u' + s)); d = abs(e[1] - b[1]); wd = np.linalg.norm(p(P, 'o' + s) - p(P, 'i' + s))
            uy = uy - m * d * _g(gx, gy, b[0], b[1], 0.8 * wd, 0.6 * d)
        return _warp(img, 0 * gx, uy)

    def intercanthal(img, P, m):
        gx, gy = grid(img); ux = 0 * gx; iod = abs(p(P, 'oR')[0] - p(P, 'oL')[0])
        for s, sign in (('R', -1), ('L', 1)):
            c = (p(P, 'o' + s) + p(P, 'i' + s)) / 2; wd = np.linalg.norm(p(P, 'o' + s) - p(P, 'i' + s))
            ux = ux + sign * m * 0.5 * iod * _g(gx, gy, c[0], c[1], 0.9 * wd, 0.7 * wd)
        return _warp(img, ux, 0 * gx)

    # sign convention: positive m => the target feature should INCREASE.  (image y is down: chin lower => larger)
    return {'nose_width': ('nose_width', {'nose_width'}, nose_width),
            'mouth_width': ('mouth_width', {'mouth_width', 'mouth_corner_tilt'}, mouth_width),
            'eye_opening': ('eye_opening', {'eye_opening'}, eye_opening),
            'jaw_width': ('jaw_width', {'jaw_width'}, jaw_width),
            'chin_height': ('chin_height', {'chin_height', 'face_height', 'face_aspect_ratio'}, chin_height),
            'brow_eye_distance': ('brow_eye_distance', {'brow_eye_distance'}, brow_height),
            'intercanthal_width': ('intercanthal_width', {'intercanthal_width', 'biocular_width'}, intercanthal)}


def controls(img: np.ndarray):
    yield 'jpeg_q60', cv2.imdecode(cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 60])[1], 1)[..., ::-1]
    h, w = img.shape[:2]; yield 'resize_0.6x', cv2.resize(cv2.resize(img, (int(w * .6), int(h * .6))), (w, h))
    yield 'brightness_0.75', np.clip(img.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    yield 'blur_sigma2', cv2.GaussianBlur(img, (0, 0), 2.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--pairs', nargs='+', required=True)
    ap.add_argument('--out', default='results'); ap.add_argument('--tta', type=int, default=16); ap.add_argument('--n-src', type=int, default=14)
    ap.add_argument('--shapes', required=True); ap.add_argument('--max-yaw', type=float, default=15); ap.add_argument('--max-pitch', type=float, default=18)
    ap.add_argument('--mags', type=float, nargs='+', default=[0.10, 0.25, 0.40]); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs, _ = load_pairs(a.pairs); files = sorted({f for p in pairs for f in p[:2]}); ident = identities(files, pairs)
    C = pickle.load(open(a.shapes, 'rb'))
    keep = {f for f, v in C.items() if abs(v['pose']['yaw']) <= a.max_yaw and abs(v['pose']['pitch']) <= a.max_pitch}   # same quality gate as the stats eval
    files = [f for f in files if f in keep]; pairs = [p for p in pairs if p[0] in keep and p[1] in keep]; ident = identities(files, pairs)
    M = {f: measure(C[f]['V']) for f in files}
    byid = collections.defaultdict(list)
    for f in files: byid[ident[f]].append(f)
    same = [(x, y) for v in byid.values() for x, y in itertools.combinations(v, 2)]
    D = np.array([difference(M[x], M[y]) for x, y in same]); sid = np.array([ident[x] for x, _ in same])
    nm_cache = {}
    def noise_for(i):
        if i not in nm_cache: nm_cache[i] = NoiseModel.fit(D[sid != i], 't')
        return nm_cache[i]

    # source images: near-frontal, one per identity first, then fill
    cand = []
    for f in files:
        r = reconstruct(Image.open(Path(a.images) / f).convert('RGB'), tta=a.tta)
        if abs(r['pose']['yaw']) < 8 and abs(r['pose']['pitch']) < 14: cand.append((f, r))
    seen, src = set(), []
    for f, r in cand:
        if ident[f] not in seen: src.append((f, r)); seen.add(ident[f])
    for f, r in cand:
        if len(src) >= a.n_src: break
        if (f, r) not in src: src.append((f, r))
    src = src[:a.n_src]; print('source images:', len(src), 'identities:', len({ident[f] for f, _ in src}))
    EDITS = make_edits(); mags = a.mags; rows = []
    # pass 1: instrument-noise controls (same photo, benign perturbations) -> d vectors per source image
    ctrl_d = collections.defaultdict(list); base = {}
    for f, r0 in src:
        img = np.array(Image.open(Path(a.images) / f).convert('RGB')); base[f] = (img, r0, measure(r0['V'][:468]))
        for cname, cimg in controls(img):
            r1 = reconstruct(Image.fromarray(cimg), tta=a.tta); ctrl_d[f].append((cname, difference(measure(r1['V'][:468]), base[f][2])))
    def inst_noise(f):                                          # cross-fitted: never calibrate on the source being tested
        D_ = np.array([d for g, L in ctrl_d.items() if g != f for _, d in L]); return NoiseModel.fit(D_, 't')
    # pass 2: edits, judged under BOTH nulls
    for f, (img, r0, m0) in base.items():
        nm_wp, nm_in = noise_for(ident[f]), inst_noise(f)
        for cname, d in ctrl_d[f]:
            rows.append(dict(src=f, edit='CONTROL:' + cname, n_flag_wp=int(bh(nm_wp.pvalues(d), 0.05).sum()), n_flag_inst=int(bh(nm_in.pvalues(d), 0.05).sum())))
        for ename, (target, allowed, fn) in EDITS.items():
            for sgn in (+1, -1):
                for m in mags:
                    try: r1 = reconstruct(Image.fromarray(fn(img, r0['P'], sgn * m)), tta=a.tta)
                    except Exception: continue
                    d = difference(measure(r1['V'][:468]), m0); k = NAMES.index(target); row = dict(src=f, edit=ename, sign=sgn, mag=m, realized=float(d[k]), dir_ok=bool(np.sign(d[k]) == sgn))
                    for tag, nm in (('wp', nm_wp), ('inst', nm_in)):
                        p_ = nm.pvalues(d); z = nm.z(d); sig = bh(p_, 0.05); flagged = {NAMES[j] for j in np.where(sig)[0]}; top = NAMES[int(np.argmax(np.abs(z)))]
                        row.update({f'detected_{tag}': bool(sig[k]), f'z_{tag}': float(z[k]), f'top1_{tag}': top, f'top1_ok_{tag}': top in allowed,
                                    f'n_flag_{tag}': len(flagged), f'n_off_{tag}': len(flagged - allowed), f'dir_ok_flagged_{tag}': (bool(np.sign(d[k]) == sgn) if sig[k] else None)})
                    rows.append(row)
    json.dump(rows, open(out / 'results_edits.json', 'w'), indent=1)
    E = [r for r in rows if not r['edit'].startswith('CONTROL')]; C = [r for r in rows if r['edit'].startswith('CONTROL')]
    print(f'{len(E)} edits, {len(C)} controls')
    print('edit | dose | n | realized/applied gain | direction ok | [instrument null] detected, top1 in coupled set, off-target flags | [within-person null] detected')
    for ename in EDITS:
        for m in mags:
            Q = [r for r in E if r['edit'] == ename and r['mag'] == m]
            if Q: print(f"{ename:18s} {m:.2f} n={len(Q):3d} gain={np.mean([q['realized']*q['sign'] for q in Q])/m:+.2f} dir={np.mean([q['dir_ok'] for q in Q]):.2f} | det_inst={np.mean([q['detected_inst'] for q in Q]):.2f} top1_inst={np.mean([q['top1_ok_inst'] for q in Q]):.2f} off_inst={np.mean([q['n_off_inst'] for q in Q]):.2f} | det_wp={np.mean([q['detected_wp'] for q in Q]):.2f}")
    print('CONTROLS flags/ctrl: instrument null %.3f ; within-person null %.3f' % (np.mean([c['n_flag_inst'] for c in C]), np.mean([c['n_flag_wp'] for c in C])))


if __name__ == '__main__':
    main()
