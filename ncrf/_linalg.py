"""Linear-algebra kernels used throughout NCRF estimation.

The Gaussian TRF basis, the inverse matrix square root used for whitening, and
the source covariance update of the Champagne iterations.
"""
from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from ._typing import FloatArray

#: Relative tolerance for treating eigenvalues and reciprocals as zero.
_R_tol = np.finfo(np.float64).eps * 1e2


def gaussian_basis(
        n_atoms: int,
        lags: npt.ArrayLike,
        basis_std: float = 0.0085,
) -> FloatArray:
    """Construct Gabor basis for the TRFs.

    Parameters
    ----------
    n_atoms
        Number of atoms; their centers are spaced evenly over ``lags``, excluding
        its endpoints.
    lags
        One-dimensional lag times covered by the basis functions,
        shape ``(n_lags,)``.
    basis_std
        Standard deviation of each Gaussian atom, expressed in the same units
        as ``lags``.

    Returns
    -------
    ndarray
        Array whose columns contain the basis atoms, shape ``(n_lags, n_atoms)``.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Using gaussian basis with {basis_std=}')
    lags = np.asarray(lags, dtype=np.float64)
    lag_start = lags[0]
    lag_stop = lags[-1]
    lag_step = (lag_stop - lag_start) / (n_atoms + 1)
    centers = np.linspace(lag_start + lag_step, lag_stop - lag_step, num=n_atoms)
    basis = np.exp(-((lags[:, None] - centers[None, :]) ** 2) / (2 * basis_std ** 2))
    return basis / basis.max()


def _inv_sqrtm(
        m: FloatArray,
        return_eig: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    e, v = linalg.eigh(m)
    e = e.real
    tol = _R_tol * e.max()
    ind = (e > tol)
    y = np.zeros((e.shape[0], 1))
    y[ind, 0] = 1 / e[ind]
    if return_eig:
        return np.sqrt(y) * v.T.conj(), np.squeeze(y[ind])
    return np.sqrt(y) * v.T.conj()


def compute_gamma(z: FloatArray, x: FloatArray) -> FloatArray:
    """Champagne update of the source covariances, batched over sources.

    For each source ``i`` computes::

        Gamma_i = Z_i**(-1/2) (Z_i**(1/2) X_i X_i' Z_i**(1/2))**(1/2) Z_i**(-1/2)
                = V E**(-1/2) U D**(1/2) U' E**(-1/2) V'

    with the eigendecompositions ``Z_i = V E V'`` and
    ``E**(1/2) V' X_i X_i' V E**(1/2) = U D U'``.

    Parameters
    ----------
    z
        Auxiliary matrices ``Z_i``, symmetric positive semi-definite, shape
        ``(n_sources, dc, dc)``.
    x
        Auxiliary coefficients ``X_i``, shape ``(n_sources, dc, rank)``.

    Returns
    -------
    ndarray
        Updated source covariances, shape ``(n_sources, dc, dc)``.
    """
    e, v = np.linalg.eigh(z)
    # eigenvalues below the tolerance (relative to each source's largest) are
    # treated as zero: dropped from E**(1/2) and excluded from its reciprocal
    significant = e > _R_tol * e.max(axis=1, keepdims=True)
    e = np.sqrt(np.where(significant, e, 0))
    xv = v.transpose(0, 2, 1) @ x
    d, u = np.linalg.eigh(e[:, :, None] * (xv @ xv.transpose(0, 2, 1)) * e[:, None, :])
    d = np.sqrt(np.clip(d, 0, None))
    e_inv = np.zeros_like(e)
    np.divide(1, e, out=e_inv, where=significant)
    w = (v * e_inv[:, None, :]) @ u
    return (w * d[:, None, :]) @ w.transpose(0, 2, 1)


def compute_gamma_diagonal(z: FloatArray, x: FloatArray) -> FloatArray:
    """Champagne update with diagonal source covariances, batched over sources.

    The CHAMP_D variant of :func:`compute_gamma` (Wipf et al., 2010, Eq. 17):
    each ``Gamma_i`` is restricted to a diagonal matrix, and the minimizer of
    ``||X_i||^2_{Gamma_i^-1} + trace(Z_i' Gamma_i)`` (Eq. 12) under that
    constraint is, for each orientation ``k``::

        Gamma_i[k, k] = ||X_i[k, :]|| / sqrt(Z_i[k, k])

    Parameters
    ----------
    z
        Auxiliary matrices ``Z_i``, symmetric positive semi-definite, shape
        ``(n_sources, dc, dc)``.
    x
        Auxiliary coefficients ``X_i``, shape ``(n_sources, dc, rank)``.

    Returns
    -------
    ndarray
        Updated (diagonal) source covariances, shape ``(n_sources, dc, dc)``.
    """
    z_diag = np.diagonal(z, axis1=1, axis2=2)  # (n_sources, dc)
    # orientations without lead-field signal (relative to each source's largest)
    # get no variance, as in compute_gamma
    significant = z_diag > _R_tol * z_diag.max(axis=1, keepdims=True)
    x_norm = np.sqrt((x ** 2).sum(axis=2))  # (n_sources, dc)
    gamma_diag = np.zeros_like(z_diag)
    np.divide(x_norm, np.sqrt(np.where(significant, z_diag, 1)), out=gamma_diag, where=significant)
    n_sources, dc = z_diag.shape
    gamma = np.zeros((n_sources, dc, dc), dtype=gamma_diag.dtype)
    idx = np.arange(dc)
    gamma[:, idx, idx] = gamma_diag
    return gamma


def compute_gamma_scalar(z: FloatArray, x: FloatArray) -> FloatArray:
    """Champagne update with isotropic source covariances, batched over sources.

    The CHAMP_S variant of :func:`compute_gamma` (Wipf et al., 2010, Eq. 17):
    each ``Gamma_i = gamma_i I`` is restricted to a scalar multiple of the
    identity, and the minimizer of ``||X_i||^2_{Gamma_i^-1} + trace(Z_i' Gamma_i)``
    (Eq. 12) under that constraint is::

        gamma_i = ||X_i||_F / sqrt(trace(Z_i))

    Parameters
    ----------
    z
        Auxiliary matrices ``Z_i``, symmetric positive semi-definite, shape
        ``(n_sources, dc, dc)``.
    x
        Auxiliary coefficients ``X_i``, shape ``(n_sources, dc, rank)``.

    Returns
    -------
    ndarray
        Updated (isotropic) source covariances, shape ``(n_sources, dc, dc)``.
    """
    z_trace = np.trace(z, axis1=1, axis2=2)  # (n_sources,)
    # sources without lead-field signal get no variance
    significant = z_trace > _R_tol * z_trace.max()
    x_norm = np.sqrt((x ** 2).sum(axis=(1, 2)))  # (n_sources,)
    gamma_i = np.zeros_like(z_trace)
    np.divide(x_norm, np.sqrt(np.where(significant, z_trace, 1)), out=gamma_i, where=significant)
    dc = z.shape[1]
    return gamma_i[:, None, None] * np.eye(dc)
