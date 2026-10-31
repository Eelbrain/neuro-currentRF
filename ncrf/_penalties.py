"""Sparsity penalties and their proximal operators used by the FASTA solver.

Each penalty comes as a pair of functions:

- the penalty itself, ``penalty(x, mu) -> float``, which returns ``mu`` times
  the (unweighted) penalty of the coefficients ``x``;
- its proximal operator, ``prox(z, mu) -> ndarray``, which returns::

      argmin_x  penalty(x, 1) + 1 / (2 * mu) ||x - z|| ** 2

The solver (``_ChampLassoState.run``) binds the penalty weight and passes both
to :class:`~ncrf._fastac.Fasta`, calling the proximal operator with the
threshold ``mu * t`` for the FASTA step size ``t``.

A new penalty must follow the same contract, in particular:

- The proximal operator must return a new array, the same shape as ``z``, and
  leave ``z`` unchanged. FASTA derives the subgradient of the penalty from the
  difference between the two (``(z - prox(z, mu * t)) / t``), which silently
  vanishes if they share a buffer.
- The coefficients have shape ``(n_sources * dc, n_columns)``, with each
  source's ``dc`` orientation components in adjacent rows (see
  ``ForwardModel.source_block``).
"""
from __future__ import annotations

import numpy as np

from ._typing import FloatArray


def g(x: FloatArray, mu: float) -> float:
    """l1 penalty, used for fixed-orientation sources.

    Computes::

        mu * sum_j |x_j|

    over all entries of ``x``.

    Parameters
    ----------
    x
        Coefficients, any shape.
    mu
        Penalty weight.

    Returns
    -------
    float
        Value of the penalty.
    """
    return mu * np.sum(np.abs(x))


def shrink(x: FloatArray, mu: float) -> FloatArray:
    """Proximal operator of the l1 penalty :func:`g` (soft thresholding).

    Computes, entry by entry::

        prox_{mu ||.||_1}(x) = argmin_z  ||z||_1 + 1 / (2 * mu) ||z - x|| ** 2
                             = sign(x) max(|x| - mu, 0)

    Parameters
    ----------
    x
        Coefficients, any shape.
    mu
        Threshold (the penalty weight times the FASTA step size).

    Returns
    -------
    ndarray
        Thresholded coefficients, a new array of the same shape as ``x`` (see
        the module docstring).
    """
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def g_group(x: FloatArray, mu: float) -> float:
    """Group (l12) penalty, used for free-orientation sources.

    Computes ``mu * gg(x)`` with::

        gg(x) = sum_{i, t} ||x_{s(i), t}||

    the sum of the Euclidean norms of the groups, where ``s(i)`` holds the rows
    ``3 * i`` to ``3 * i + 2`` of source ``i``'s three orientation components and
    ``t`` runs over the columns of ``x``. The penalty thus favours sources that
    are inactive in all orientations at once, but does not favour any
    orientation over the others.

    Parameters
    ----------
    x
        Coefficients, shape ``(3 * n_sources, n_columns)``, with each source's
        three orientation components in adjacent rows (see
        ``ForwardModel.source_block``). The groups are formed on a reshaped
        view, so ``x`` is not modified.
    mu
        Penalty weight.

    Returns
    -------
    float
        Value of the penalty.
    """
    l = x.shape[1]
    x3 = x.reshape(-1, 3, l)
    return mu * np.sqrt((x3 ** 2).sum(axis=1)).sum()


def proxg_group(z: FloatArray, mu: float) -> FloatArray:
    r"""Proximal operator of the group (l12) penalty :func:`g_group`.

    Computes::

        prox_{mu gg}(z) = argmin_x  gg(x) + 1 / (2 * mu) ||x - z|| ** 2

    which is group-wise soft thresholding::

        x_s = max(1 - mu / ||z_s||, 0) z_s

    A group ``s`` is the three orientation components of one source in one
    column of ``z``, so each column is thresholded separately.

    Parameters
    ----------
    z
        Coefficients, shape ``(3 * n_sources, n_columns)``, with each source's
        three orientation components in adjacent rows (see
        ``ForwardModel.source_block``).
    mu
        Threshold (the penalty weight times the FASTA step size).

    Returns
    -------
    ndarray
        Thresholded coefficients, a new array of the same shape as ``z`` (see
        the module docstring).
    """
    z3 = z.reshape(-1, 3, z.shape[1])
    norm = np.sqrt((z3 ** 2).sum(axis=1, keepdims=True))
    # mu / ||z_s|| for the groups that survive the shrinkage, 1 (a factor of 0) for the rest
    ratio = np.ones_like(norm)
    np.divide(mu, norm, out=ratio, where=norm > mu)
    return ((1 - ratio) * z3).reshape(z.shape)
