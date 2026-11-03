"""Offline tests for the OT matcher (no model weights needed)."""
import numpy as np
import ot
import pytest
import torch

from geco.matching import _solve_sinkhorn, get_confidence_aware_marginals, match_from_features


def _feats(h=8, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, d, h, h, generator=g)


def test_marginals_sum_to_one_and_dustbin_is_one_patch():
    c = torch.rand(8, 8)
    mu, nu = get_confidence_aware_marginals(c, c)
    assert mu.shape == (65,) and nu.shape == (65,)
    assert mu.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert mu[-1].item() == pytest.approx(mu[:-1].mean().item(), rel=0.2)


def test_mass_argument_is_noop():
    c = torch.rand(6, 6)
    a, _ = get_confidence_aware_marginals(c, c, mass=0.9)
    b, _ = get_confidence_aware_marginals(c, c, mass=0.3)
    assert torch.allclose(a, b, atol=1e-6)


def test_plan_has_correct_marginals():
    f = _feats()
    r = match_from_features(f, f)
    plan = r.transport_plan
    assert torch.isfinite(plan).all()
    assert plan.sum().item() == pytest.approx(1.0, abs=1e-3)
    assert plan.shape == (65, 65)


def test_identical_features_match_diagonal():
    f = _feats(h=8, seed=1)
    r = match_from_features(f, f)
    pred = r.transport_plan[:-1, :-1].argmax(dim=1)
    assert (pred == torch.arange(64)).float().mean().item() > 0.9


def test_solver_is_shift_invariant_and_finite_for_tiny_reg():
    n = 30
    rng = np.random.default_rng(0)
    cost = torch.tensor(1.0 + rng.random((n, n)), dtype=torch.float32)
    mu = torch.ones(n) / n
    p1 = _solve_sinkhorn(mu, mu, cost, reg=0.01)
    p2 = _solve_sinkhorn(mu, mu, cost + 5.0, reg=0.01)
    assert torch.isfinite(p1).all()
    assert torch.allclose(p1, p2, atol=1e-5)
    ref = ot.sinkhorn(mu.double().numpy(), mu.double().numpy(), (cost - cost.min()).double().numpy(), 0.01, method="sinkhorn_log")
    assert np.abs(p1.numpy() - ref).max() < 1e-4


def test_low_confidence_patches_are_not_forced_into_matches():
    f = _feats(seed=3)
    g = _feats(seed=4)  # unrelated target
    related = match_from_features(f, f).mean_confidence
    unrelated = match_from_features(f, g).mean_confidence
    assert related > unrelated
