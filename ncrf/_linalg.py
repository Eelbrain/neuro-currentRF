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
