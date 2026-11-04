import numpy as np
import pytest

from geco.calibration import auc, best_threshold, calibrate, eer, load_calibration, save_calibration
from geco.face import GEOMETRY_WEIGHT, verdict_from_margin


def _pairs(n=400, same=(0.97, 0.01), diff=(0.90, 0.02), seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2 == 0
    s = np.where(y, rng.normal(*same, n), rng.normal(*diff, n))
    return s, y


def test_auc_perfect_and_chance():
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 0, 0])) == 0.5


def test_threshold_found_in_narrow_cone_not_clipped_to_grid():
    s, y = _pairs()
    thr, acc = best_threshold(s, y)
    assert 0.91 < thr < 0.96 and acc > 0.95  # old fixed 0.05-step grid could not resolve this


def test_calibrate_reports_heldout_and_roundtrips(tmp_path):
    s, y = _pairs()
    rep = calibrate(s, y)
    assert rep["auc"] > 0.97 and rep["eer"] < 0.05
    assert abs(rep["heldout_accuracy"] - rep["in_sample_accuracy"]) < 0.05
    p = tmp_path / "face_calibration.json"
    save_calibration(rep, p)
    assert load_calibration(p)["threshold"] == pytest.approx(rep["threshold"])
    assert load_calibration(tmp_path / "missing.json") is None


def test_calibrate_rejects_degenerate_input():
    with pytest.raises(ValueError):
        calibrate(np.random.rand(30), np.ones(30, bool))


def test_eer_chance_level():
    rng = np.random.default_rng(1)
    assert 0.4 < eer(rng.random(2000), rng.random(2000) > 0.5) < 0.6


def test_verdict_bands_match_legacy_at_default_scale():
    assert verdict_from_margin(0.03)[0] == "confident_match"
    assert verdict_from_margin(0.01)[0] == "likely_match"
    assert verdict_from_margin(0.0)[0] == "ambiguous"
    assert verdict_from_margin(-0.01)[0] == "likely_different"
    assert verdict_from_margin(-0.03)[0] == "confident_different"
    assert verdict_from_margin(0.2)[1] == 99.9


def test_verdict_bands_scale_with_margin_scale():
    assert verdict_from_margin(0.03, margin_scale=0.02)[0] == "confident_match"
    assert verdict_from_margin(0.05, margin_scale=0.2)[0] == "likely_match"


def test_geometry_does_not_shift_decision_score():
    assert GEOMETRY_WEIGHT == 0.0
