"""Threshold calibration for verification scores (no torch dependency).

Picks the decision threshold on one half of the labelled pairs and reports
accuracy on the OTHER half, so the headline number is not tuned on its own test
set. Also reports AUC and EER, which do not depend on any threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CALIBRATION_FILENAME = "face_calibration.json"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (probability a same-person pair outscores a different one)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, step: float = 0.005):
    """Accuracy at every threshold across the OBSERVED score range (not a fixed -1..1 grid)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    lo, hi = scores.min() - step, scores.max() + step
    grid = np.arange(lo, hi + step, step)
    acc = np.array([((scores >= t) == labels).mean() for t in grid])
    return grid, acc


def best_threshold(scores: np.ndarray, labels: np.ndarray, step: float = 0.005) -> tuple[float, float]:
    """Threshold with max accuracy; ties resolved to the middle of the best plateau."""
    grid, acc = threshold_sweep(scores, labels, step)
    best = acc.max()
    idx = np.flatnonzero(acc >= best - 1e-12)
    return float(grid[idx[len(idx) // 2]]), float(best)


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    grid, _ = threshold_sweep(scores, labels, 0.002)
    far = np.array([(scores[~labels] >= t).mean() for t in grid])
    frr = np.array([(scores[labels] < t).mean() for t in grid])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def calibrate(scores, labels, step: float = 0.005) -> dict:
    """Held-out calibration: threshold from even-indexed pairs, accuracy on odd-indexed."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if len(s) < 20 or y.all() or (~y).all():
        raise ValueError("Need >= 20 pairs with both same and different labels.")
    fit, test = np.arange(len(s)) % 2 == 0, np.arange(len(s)) % 2 == 1
    thr, fit_acc = best_threshold(s[fit], y[fit], step)
    held_acc = float(((s[test] >= thr) == y[test]).mean())
    all_thr, all_acc = best_threshold(s, y, step)
    return {
        "threshold": all_thr,  # final deployed value: fit on all pairs
        "heldout_threshold": thr,
        "fit_accuracy": fit_acc,
        "heldout_accuracy": held_acc,
        "in_sample_accuracy": all_acc,
        "auc": auc(s, y),
        "eer": eer(s, y),
        "same_mean": float(s[y].mean()),
        "same_std": float(s[y].std()),
        "diff_mean": float(s[~y].mean()),
        "diff_std": float(s[~y].std()),
        "n_pairs": int(len(s)),
        # Width over which a verdict is "ambiguous": ~1 pooled std of the two classes.
        "margin_scale": float(max(1e-3, np.sqrt((s[y].var() + s[~y].var()) / 2))),
    }


def save_calibration(report: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))


def load_calibration(path: Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        float(d["threshold"])
        return d
    except Exception:
        return None
