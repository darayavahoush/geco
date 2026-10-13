"""Feature-level 3D face difference attribution.

Given two pose-normalised 468-vertex face shapes (``face_mp3d.reconstruct``), report WHICH anatomical
feature differs, in WHICH direction, by HOW MUCH, and whether that difference is larger than the variation a single
person shows between photographs (calibrated, leave-identity-out, Benjamini-Hochberg corrected).

Design choices
  * Scale-free: every measurement is divided by the centroid size of the 468 landmarks (geometric morphometrics).
  * Length features are compared as log-ratios (symmetric: A/B and B/A give equal magnitude, opposite sign),
    angles as degree differences.
  * z = d / sigma_within, with sigma_within = robust std (1.4826*MAD) of d over SAME-PERSON photo pairs.
  * Depth-derived features (use the network's z estimate) are flagged ``depth=True`` and should be read with
    lower confidence: z is a learned prior, not a measurement.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ------------------------------------------------------------------ landmark indices (MediaPipe canonical)
I = dict(oR=33, iR=133, oL=263, iL=362, uR=159, lR=145, uL=386, lL=374, bR=105, bL=334, nas=168, sub=2, tip=1,
         alR=129, alL=358, mR=61, mL=291, lu_t=0, lu_b=13, ll_t=14, ll_b=17, chin=152, top=10,
         zR=234, zL=454, gR=172, gL=397)


REF_WIDTH = 0.123   # mean centroid-normalised bizygomatic width on the pilot set; unit for 'lin' features


def _cs(V):  # centroid size
    return float(np.sqrt(((V - V.mean(0)) ** 2).sum()))


def _w(V, a, b):  # horizontal distance
    return abs(V[I[a], 0] - V[I[b], 0])


def _h(V, a, b):  # vertical distance (a above b)
    return V[I[a], 1] - V[I[b], 1]


def _ang(V, a, b):  # tilt in degrees of the segment a->b, mirrored so that right/left eyes are comparable
    d = V[I[b]] - V[I[a]]
    return float(np.degrees(np.arctan2(d[1], abs(d[0]))))


@dataclass(frozen=True)
class Feature:
    name: str
    region: str
    kind: str          # "ratio" (log-ratio) | "lin" (difference as fraction of mean face width; for tiny quantities) | "angle" (deg)
    fn: callable
    up: str            # word for larger value (used in sentences)
    down: str
    depth: bool = False


def _mean2(f):
    return lambda V: 0.5 * (f(V, 'R') + f(V, 'L'))


FEATURES: list[Feature] = [
    Feature('face_width', 'face', 'ratio', lambda V: _w(V, 'zR', 'zL'), 'wider', 'narrower'),
    Feature('face_height', 'face', 'ratio', lambda V: _h(V, 'nas', 'chin'), 'longer', 'shorter'),   # nasion-gnathion (forehead-top landmark 10 is hair-dependent)
    Feature('jaw_width', 'jaw', 'ratio', lambda V: _w(V, 'gR', 'gL'), 'wider', 'narrower'),
    Feature('chin_height', 'jaw', 'ratio', lambda V: _h(V, 'll_b', 'chin'), 'taller', 'shorter'),
    Feature('eye_width', 'eyes', 'ratio', _mean2(lambda V, s: _w(V, 'o' + s, 'i' + s)), 'wider', 'narrower'),
    Feature('eye_opening', 'eyes', 'lin', _mean2(lambda V, s: _h(V, 'u' + s, 'l' + s)), 'more open', 'less open'),
    Feature('intercanthal_width', 'eyes', 'ratio', lambda V: _w(V, 'iR', 'iL'), 'wider-set', 'closer-set'),
    Feature('biocular_width', 'eyes', 'ratio', lambda V: _w(V, 'oR', 'oL'), 'wider-set', 'closer-set'),
    Feature('canthal_tilt', 'eyes', 'angle', _mean2(lambda V, s: _ang(V, 'i' + s, 'o' + s)), 'more upslanted', 'more downslanted'),
    Feature('brow_eye_distance', 'brows', 'lin', _mean2(lambda V, s: _h(V, 'b' + s, 'u' + s)), 'higher above the eye', 'closer to the eye'),
    Feature('nose_width', 'nose', 'ratio', lambda V: _w(V, 'alR', 'alL'), 'wider', 'narrower'),
    Feature('nose_length', 'nose', 'ratio', lambda V: _h(V, 'nas', 'sub'), 'longer', 'shorter'),
    Feature('nose_projection', 'nose', 'lin',
            lambda V: V[I['tip'], 2] - 0.5 * (V[I['alR'], 2] + V[I['alL'], 2]), "more projecting", 'flatter', True),
    Feature('mouth_width', 'mouth', 'ratio', lambda V: _w(V, 'mR', 'mL'), 'wider', 'narrower'),
    Feature('upper_lip_thickness', 'mouth', 'lin', lambda V: _h(V, 'lu_t', 'lu_b'), 'thicker', 'thinner'),
    Feature('lower_lip_thickness', 'mouth', 'lin', lambda V: _h(V, 'll_t', 'll_b'), 'thicker', 'thinner'),
    Feature('philtrum_length', 'mouth', 'lin', lambda V: _h(V, 'sub', 'lu_t'), 'longer', 'shorter'),
    Feature('mouth_corner_tilt', 'mouth', 'angle',
            lambda V: float(np.degrees(np.arctan2(0.5 * (V[I['mR'], 1] + V[I['mL'], 1]) - V[I['lu_b'], 1],
                                                  abs(V[I['mL'], 0] - V[I['mR'], 0]) / 2))), 'more upturned', 'more downturned'),
    Feature('face_aspect_ratio', 'face', 'ratio', lambda V: _h(V, 'nas', 'chin') / max(_w(V, 'zR', 'zL'), 1e-6), 'longer-faced', 'rounder-faced'),
]
NAMES = [f.name for f in FEATURES]

# Test-retest reliability ICC(1,1) of each feature across different photos of one person (pilot: 13 identities, 57 pose-gated
# images; see docs/FACE_DIFFERENCE_PAPER.md).  Re-estimate on your own data with scripts/eval/evaluate_face_diff.py.
RELIABILITY_ICC = dict(face_width=.59, face_height=.25, jaw_width=.54, chin_height=.33, eye_width=.13, eye_opening=.17,
                       intercanthal_width=.24, biocular_width=.18, canthal_tilt=.59, brow_eye_distance=.59, nose_width=.27,
                       nose_length=.06, nose_projection=.23, mouth_width=.32, upper_lip_thickness=.28, lower_lip_thickness=.37,
                       philtrum_length=.40, mouth_corner_tilt=.34, face_aspect_ratio=.36)
MIN_ICC = 0.30     # features below this are never asserted as "different": photo-to-photo noise swamps person-to-person signal

# ------------------------------------------------------------------ regions (geometric partition of the canonical mesh)
def _regions(canon: np.ndarray) -> dict[str, np.ndarray]:
    x, y = np.abs(canon[:, 0]), canon[:, 1]
    r = {}
    taken = np.zeros(len(canon), bool)
    for name, m in [('eyes', (y > 0.02) & (y < 0.075) & (x > 0.045) & (x < 0.16)),
                    ('brows', (y >= 0.075) & (y < 0.17) & (x < 0.19)),
                    ('mouth', (y > -0.225) & (y < -0.115) & (x < 0.105)),
                    ('nose', (y > -0.115) & (y <= 0.02) & (x < 0.07)),
                    ('jaw_chin', (y <= -0.23) | ((x > 0.14) & (y < -0.12))),
                    ('forehead', y >= 0.17),
                    ('cheeks', (y > -0.23) & (y < 0.02) & (x >= 0.07))]:
        m = m & ~taken; r[name] = np.where(m)[0]; taken |= m
    return r


# ------------------------------------------------------------------ measurement + comparison
def measure(V: np.ndarray) -> np.ndarray:
    """Feature vector for a [468,3] pose-normalised shape (scale-free)."""
    V = np.asarray(V, np.float64)[:468]; V = V / _cs(V)
    return np.array([f.fn(V) for f in FEATURES], np.float64)


def difference(ma: np.ndarray, mb: np.ndarray) -> np.ndarray:
    """Signed difference d = A - B on each feature's natural scale (log-ratio or degrees)."""
    d = np.zeros(len(FEATURES))
    for k, f in enumerate(FEATURES):
        d[k] = (np.log(max(ma[k], 1e-9) / max(mb[k], 1e-9)) if f.kind == 'ratio' else (ma[k] - mb[k]) / REF_WIDTH if f.kind == 'lin' else ma[k] - mb[k])
    return d


def robust_sigma(d: np.ndarray) -> np.ndarray:
    """1.4826*MAD per column (centre at 0: the same-person difference distribution is symmetric by construction)."""
    return 1.4826 * np.median(np.abs(d), axis=0) + 1e-9


def _norm_sf(z):
    from math import erfc, sqrt
    return erfc(abs(z) / sqrt(2.0))


class NoiseModel:
    """Within-person variation of each feature difference d (same person, different photos).

    kind='gauss': zero-mean normal with robust sigma.  kind='t': zero-mean Student-t (df, scale fitted per feature by
    maximum likelihood) -- same-person differences are heavy-tailed (expression, makeup, lighting), so a normal
    model under-states tail probability and over-flags.  ``z`` is the normal-equivalent score of the two-sided p-value.
    """
    def __init__(self, df: np.ndarray, scale: np.ndarray):
        self.df, self.scale = np.asarray(df, float), np.asarray(scale, float)

    @classmethod
    def fit(cls, D: np.ndarray, kind: str = 't') -> 'NoiseModel':
        D = np.vstack([D, -D])                                   # pair order is arbitrary -> symmetric
        if kind == 'gauss':
            return cls(np.full(D.shape[1], np.inf), robust_sigma(D))
        from scipy import stats
        df, sc = [], []
        for k in range(D.shape[1]):
            d_, _, s_ = stats.t.fit(D[:, k], floc=0.0)
            df.append(float(np.clip(d_, 1.5, 200.0))); sc.append(float(s_))
        return cls(np.array(df), np.array(sc))

    def pvalues(self, d: np.ndarray) -> np.ndarray:
        from scipy import stats
        d = np.asarray(d, float); p = np.empty_like(d)
        for k in range(len(d)):
            p[k] = (2 * stats.norm.sf(abs(d[k]) / self.scale[k]) if np.isinf(self.df[k])
                    else 2 * stats.t.sf(abs(d[k]), self.df[k], scale=self.scale[k]))
        return np.clip(p, 1e-300, 1.0)

    def z(self, d: np.ndarray) -> np.ndarray:
        from scipy import stats
        return np.sign(d) * stats.norm.isf(self.pvalues(d) / 2)

    def threshold(self, alpha: float = 0.01) -> np.ndarray:
        """Smallest |d| that is significant at two-sided ``alpha`` (the minimum reliably detectable difference)."""
        from scipy import stats
        return np.array([stats.norm.isf(alpha / 2) * s if np.isinf(df) else stats.t.isf(alpha / 2, df) * s
                         for df, s in zip(self.df, self.scale)])


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of rejected hypotheses."""
    n = len(p); o = np.argsort(p); thr = q * (np.arange(1, n + 1) / n); ok = p[o] <= thr
    out = np.zeros(n, bool)
    if ok.any(): out[o[:np.max(np.where(ok)[0]) + 1]] = True
    return out


def procrustes_displacement(Va: np.ndarray, Vb: np.ndarray):
    """Similarity-align A onto B (3D, 468 pts); return per-vertex displacement vectors / centroid size and RMS."""
    A = np.asarray(Va, np.float64)[:468]; B = np.asarray(Vb, np.float64)[:468]
    A = (A - A.mean(0)) / _cs(A); B = (B - B.mean(0)) / _cs(B)
    U, s, Vt = np.linalg.svd(B.T @ A); d = np.sign(np.linalg.det(U @ Vt)); E = np.diag([1, 1, d]); R = U @ E @ Vt
    sc = (s * np.diag(E)).sum() / (A ** 2).sum(); disp = B - sc * (A @ R.T)
    return disp, float(np.sqrt((disp ** 2).sum(1).mean()))


def region_rms(disp: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(np.sqrt((disp[v] ** 2).sum(1).mean())) for k, v in regions.items() if len(v)}


@dataclass
class FeatureDiff:
    name: str; region: str; value_a: float; value_b: float; delta: float; unit: str; z: float; q_significant: bool
    depth: bool; sentence: str; icc: float = float('nan'); evidence: str = 'within normal variation'


def compare(Va, Vb, noise, q: float = 0.05, label_a: str = 'A', label_b: str = 'B') -> list[FeatureDiff]:
    """Full per-feature comparison. ``noise`` = ``NoiseModel`` fitted on same-person photo pairs (an ndarray is
    interpreted as per-feature Gaussian sigma)."""
    if not isinstance(noise, NoiseModel): noise = NoiseModel(np.full(len(FEATURES), np.inf), np.asarray(noise, float))
    ma, mb = measure(Va), measure(Vb); d = difference(ma, mb); p = noise.pvalues(d); z = noise.z(d)
    sig = bh(p, q)
    out = []
    for k, f in enumerate(FEATURES):
        if f.kind == 'ratio':
            pct = 100 * (np.exp(d[k]) - 1); unit = '%'; mag = f'{abs(pct):.1f}%'; delta = float(pct)
        elif f.kind == 'lin':
            unit = '% of face width'; mag = f'{abs(100 * d[k]):.1f}% of face width'; delta = float(100 * d[k])
        else:
            unit = 'deg'; mag = f'{abs(d[k]):.1f} deg'; delta = float(d[k])
        word = f.up if d[k] > 0 else f.down
        s = f"{f.name.replace('_', ' ')}: {label_a} is {word} than {label_b} by {mag}"
        icc = RELIABILITY_ICC.get(f.name, float('nan'))
        ok = icc >= MIN_ICC
        ev = ('significant' if (sig[k] and ok) else 'flagged but unreliable' if sig[k] else 'suggestive' if (p[k] < 0.05 and ok) else 'within normal variation')
        out.append(FeatureDiff(f.name, f.region, float(ma[k]), float(mb[k]), delta, unit, float(z[k]), bool(sig[k]), f.depth, s, icc, ev))
    return sorted(out, key=lambda r: -abs(r.z))


def explain(diffs: list[FeatureDiff], top: int = 6) -> str:
    """Plain-language report. A difference is only ASSERTED if it is statistically significant against within-person
    photo-to-photo variation AND the feature is reliable (ICC >= MIN_ICC); anything else is reported as such."""
    sig = [r for r in diffs if r.evidence == 'significant'][:top]; sug = [r for r in diffs if r.evidence == 'suggestive'][:top]
    unrel = [r for r in diffs if r.evidence == 'flagged but unreliable']; low = sorted(r.name.replace('_', ' ') for r in diffs if not (r.icc >= MIN_ICC))
    L = []
    if sig: L += ["Differences larger than between-photo variation of one person (FDR q<0.05, reliable features):"] + [f"  - {r.sentence} (z={r.z:+.1f}, ICC={r.icc:.2f}{', depth-derived' if r.depth else ''})" for r in sig]
    if sug: L += ["Suggestive only (nominal p<0.05, uncorrected):"] + [f"  - {r.sentence} (z={r.z:+.1f}, ICC={r.icc:.2f})" for r in sug]
    if not sig and not sug: L += ["No reliable feature differs by more than expected between two photos of the same person."]
    if unrel: L += ["Exceeded the threshold but NOT asserted (unreliable feature; likely landmark noise): " + ", ".join(r.name.replace('_', ' ') for r in unrel) + "."]
    L += [f"Not assessable from single photos (ICC<{MIN_ICC}): " + ", ".join(low) + "."]
    return "\n".join(L)
