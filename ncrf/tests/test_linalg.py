"""Tests for the linear-algebra kernels."""

import numpy as np
import pytest
from scipy import linalg, optimize

from ncrf._linalg import (
    _R_tol,
    compute_gamma,
    compute_gamma_diagonal,
    compute_gamma_scalar,
)


def _compute_gamma_reference(z, x):
    """Gamma = Z**(-1/2) (Z**(1/2) X X' Z**(1/2))**(1/2) Z**(-1/2) for one source, by matrix functions."""
    e, v = linalg.eigh(z)
    keep = e > _R_tol * e.max()
    z_sqrt = (v[:, keep] * np.sqrt(e[keep])) @ v[:, keep].T
    z_inv_sqrt = (v[:, keep] / np.sqrt(e[keep])) @ v[:, keep].T
    m = z_sqrt @ x @ x.T @ z_sqrt
    d, u = linalg.eigh(m)
    m_sqrt = (u * np.sqrt(np.clip(d, 0, None))) @ u.T
    return z_inv_sqrt @ m_sqrt @ z_inv_sqrt


@pytest.mark.parametrize('dc', [1, 3])
def test_compute_gamma_matches_per_source_formula(dc):
    rng = np.random.default_rng(0)
    n_sources, rank = 20, 7
    lead_field = rng.normal(size=(n_sources, 15, dc))
    z = lead_field.transpose(0, 2, 1) @ lead_field
    x = rng.normal(size=(n_sources, dc, rank))

    gamma = compute_gamma(z, x)

    assert gamma.shape == (n_sources, dc, dc)
    for i in range(n_sources):
        np.testing.assert_allclose(gamma[i], _compute_gamma_reference(z[i], x[i]), atol=1e-12)
    if dc == 1:
        # fixed orientation: Gamma_i = ||x_i|| / sqrt(z_i)
        np.testing.assert_allclose(gamma[:, 0, 0], np.sqrt((x ** 2).sum(axis=(1, 2)) / z[:, 0, 0]))


def test_compute_gamma_rank_deficient_source():
    """Directions in which a source's lead field vanishes get no variance.

    The Gram matrix of collinear lead-field columns is singular up to rounding
    noise in its eigenvalues (here made explicit, at a relative 1e-17); inverting
    that noise would blow up Gamma by orders of magnitude.
    """
    rng = np.random.default_rng(1)
    orientation = np.array([1.0, 2.0, -0.5])
    lead_field = rng.normal(size=(100, 1)) * orientation
    z = lead_field.T @ lead_field
    z += 1e-17 * z.max() * np.eye(3)
    x = rng.normal(size=(3, 7))

    gamma = compute_gamma(z[None], x[None])[0]

    e, v = linalg.eigh(gamma)
    np.testing.assert_allclose(e[:2], 0, atol=1e-12 * e[2])
    np.testing.assert_allclose(np.abs(v[:, 2]), np.abs(orientation) / linalg.norm(orientation))
    np.testing.assert_allclose(gamma, _compute_gamma_reference(z, x), atol=1e-12)
    # a source with no lead field at all has zero covariance
    np.testing.assert_array_equal(compute_gamma(np.zeros((1, 3, 3)), x[None]), 0)


def _champagne_cost(gamma, z, x):
    """Per-source Champagne cost (Wipf et al., 2010, Eq. 12): ||X||^2_{Gamma^-1} + trace(Z' Gamma)."""
    return np.trace(x.T @ np.linalg.solve(gamma, x)) + np.trace(z.T @ gamma)


def _random_sources(dc, seed=0, n_sources=10, rank=7):
    rng = np.random.default_rng(seed)
    lead_field = rng.normal(size=(n_sources, 15, dc))
    z = lead_field.transpose(0, 2, 1) @ lead_field
    x = rng.normal(size=(n_sources, dc, rank))
    return z, x


@pytest.mark.parametrize('dc', [1, 3])
def test_compute_gamma_diagonal_minimizes_cost(dc):
    """CHAMP_D: the update is the minimizer of Eq. 12 over diagonal Gamma."""
    z, x = _random_sources(dc)

    gamma = compute_gamma_diagonal(z, x)

    assert gamma.shape == (len(z), dc, dc)
    off_diagonal = ~np.eye(dc, dtype=bool)
    np.testing.assert_array_equal(gamma[:, off_diagonal], 0)
    for i in range(len(z)):
        def cost(log_g, z_i=z[i], x_i=x[i]):
            return _champagne_cost(np.diag(np.exp(log_g)), z_i, x_i)
        res = optimize.minimize(cost, np.zeros(dc), method='BFGS', options={'gtol': 1e-10})
        np.testing.assert_allclose(np.diag(gamma[i]), np.exp(res.x), rtol=1e-5)
        assert _champagne_cost(gamma[i], z[i], x[i]) <= res.fun + 1e-10


@pytest.mark.parametrize('dc', [1, 3])
def test_compute_gamma_scalar_minimizes_cost(dc):
    """CHAMP_S: the update is the minimizer of Eq. 12 over Gamma = gamma I."""
    z, x = _random_sources(dc)

    gamma = compute_gamma_scalar(z, x)

    assert gamma.shape == (len(z), dc, dc)
    for i in range(len(z)):
        np.testing.assert_array_equal(gamma[i], gamma[i, 0, 0] * np.eye(dc))
        def cost(log_g, z_i=z[i], x_i=x[i]):
            return _champagne_cost(np.exp(log_g) * np.eye(dc), z_i, x_i)
        res = optimize.minimize_scalar(cost)
        np.testing.assert_allclose(gamma[i, 0, 0], np.exp(res.x), rtol=1e-5)
        assert _champagne_cost(gamma[i], z[i], x[i]) <= res.fun + 1e-10


def test_gamma_updates_agree_for_fixed_orientation():
    """With dc = 1 the three parameterizations coincide."""
    z, x = _random_sources(1, seed=2)

    gamma = compute_gamma(z, x)

    np.testing.assert_allclose(compute_gamma_diagonal(z, x), gamma, rtol=1e-12)
    np.testing.assert_allclose(compute_gamma_scalar(z, x), gamma, rtol=1e-12)


def test_restricted_gamma_updates_cost_ordering():
    """Fewer constraints can only lower the per-source cost: full <= diagonal <= scalar."""
    z, x = _random_sources(3, seed=3)

    full, diagonal, scalar = (f(z, x) for f in (compute_gamma, compute_gamma_diagonal, compute_gamma_scalar))

    for i in range(len(z)):
        cost_full, cost_diagonal, cost_scalar = (_champagne_cost(g[i], z[i], x[i]) for g in (full, diagonal, scalar))
        assert cost_full <= cost_diagonal + 1e-10
        assert cost_diagonal <= cost_scalar + 1e-10


@pytest.mark.parametrize('update', [compute_gamma_diagonal, compute_gamma_scalar])
def test_restricted_gamma_updates_no_lead_field(update):
    """Sources (diagonal: orientations) without lead field get no variance, without dividing by zero."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 3, 7))
    z = np.zeros((2, 3, 3))
    z[1] = np.diag([2.0, 0.0, 1.0])

    with np.errstate(all='raise'):
        gamma = update(z, x)

    assert np.isfinite(gamma).all()
    np.testing.assert_array_equal(gamma[0], 0)
    if update is compute_gamma_diagonal:
        assert gamma[1, 1, 1] == 0
        assert gamma[1, 0, 0] > 0 and gamma[1, 2, 2] > 0
