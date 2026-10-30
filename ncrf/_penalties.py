"""Sparsity penalties and their proximal operators used by the FASTA solver."""
from __future__ import annotations

import numpy as np

from ._typing import FloatArray


def g(x: FloatArray, mu: float) -> float:
    """Vector l1-norm penalty."""
    return mu * np.sum(np.abs(x))


def shrink(x: FloatArray, mu: float) -> FloatArray:
    """Soft-thresholding operator."""
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def g_group(x: FloatArray, mu: float) -> float:
    r"""group (l12) norm penalty:

            gg(x) = \sum ||x_s_{i,t}||

    where s_{i,t} = {x_{j,t}: j = 1*dc:(i+1)*dc}, i \in {1,2,...,#sources}, t \in {1,2,...,M}

    The three orientation components per source (fixed ``dc == 3``) are grouped
    along a reshaped view, so the caller's array is not modified.
    """
    l = x.shape[1]
    x3 = x.reshape(-1, 3, l)
    return mu * np.sqrt((x3 ** 2).sum(axis=1)).sum()


def proxg_group(z: FloatArray, mu: float) -> FloatArray:
    """proximal operator for gg(x):

            prox_{mu gg}(x) = min  gg(z) + 1/ (2 * mu) ||x-z|| ** 2
                    x_s = max(1 - mu/||z_s||, 0) z_s

    The three orientation components per source (fixed ``dc == 3``) are grouped
    along a reshaped view. Like :func:`shrink`, the result is a new array and ``z``
    is left unchanged; FASTA derives the subgradient from the difference between
    the two, which vanishes if they share a buffer.
    """
    z3 = z.reshape(-1, 3, z.shape[1])
    norm = np.sqrt((z3 ** 2).sum(axis=1, keepdims=True))
    # mu / ||z_s|| for the groups that survive the shrinkage, 1 (a factor of 0) for the rest
    ratio = np.ones_like(norm)
    np.divide(mu, norm, out=ratio, where=norm > mu)
    return ((1 - ratio) * z3).reshape(z.shape)
