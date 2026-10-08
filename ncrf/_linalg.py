"""Linear-algebra kernels used throughout NCRF estimation.

Small numeric helpers (tolerance-aware inverses, matrix square roots) and the
per-source covariance updates used by the Champagne iterations.
"""
from __future__ import annotations

import logging
from math import sqrt

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .dsyevh3C import compute_gamma_c
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


def _myinv(x: FloatArray) -> FloatArray:
    """Compute a tolerance-aware elementwise reciprocal."""
    x = np.real(np.array(x))
    tol = _R_tol * x.max()
    ind = (x > tol)
    y = np.zeros(x.shape)
    y[ind] = 1 / x[ind]
    return y


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


def _compute_gamma_i(z: FloatArray, x: FloatArray) -> FloatArray:
    """Computes Gamma_i

    Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
           = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
           = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
           = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    Parameters
    ----------
    z
        Auxiliary matrix for one source block.
    x
        Auxiliary coefficients for the same source block.

    Returns
    -------
    ndarray
        Updated block covariance matrix.
    """
    [e, v] = linalg.eig(z)
    e = e.real
    e[e < 0] = 0
    temp = np.dot(x.T, v)
    temp = np.real(np.dot(temp.conj().T, temp))
    e = np.sqrt(e)
    [d, u] = linalg.eig((temp * e) * e[:, np.newaxis])
    d = d.real
    d[d < 0] = 0
    d = np.sqrt(d)
    temp = np.dot(v * _myinv(np.real(e)), u)
    return np.array(np.real(np.dot(temp * d, temp.conj().T)))


def _compute_gamma_ip(z: FloatArray, x: FloatArray, gamma: FloatArray) -> None:
    """Wrapper function of Cython function 'compute_gamma_c'

    Computes Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
                   = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
                   = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
                   = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    Parameters
    ----------
    z
        Auxiliary square matrix for one source block, usually of shape ``(dc, dc)``.
    x
        Auxiliary coefficients for the same source block.
    gamma
        Output array that is updated in place with the new block covariance.
    """
    assert x.shape[0] == 3
    a = np.dot(x, x.T)
    compute_gamma_c(z, a, gamma)
    return


def compute_gamma(z: FloatArray, x: FloatArray, dc: int) -> FloatArray | float:
    """Return the updated source-block covariance for one Champagne step.

    Wraps the per-orientation update paths behind a single return-value
    interface: a scalar for fixed orientation (``dc == 1``), the Cython kernel
    for the three-component free-orientation case (``dc == 3``), and the general
    eigendecomposition otherwise.
    """
    if dc == 1:
        return sqrt((x ** 2).sum()) / np.real(sqrt(z))
    elif dc == 3:
        gamma = np.empty((3, 3))
        _compute_gamma_ip(z, x, gamma)
        return gamma
    else:
        return _compute_gamma_i(z, x)
