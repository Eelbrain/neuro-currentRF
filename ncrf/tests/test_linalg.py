"""Tests for the linear-algebra kernels."""

import numpy as np
import pytest
from scipy import linalg

from ncrf._linalg import _R_tol, compute_gamma


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
