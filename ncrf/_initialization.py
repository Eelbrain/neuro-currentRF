"""Empirical-Bayes / MNE-style initialization for the NCRF source covariance."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
from scipy import linalg

from ._typing import FloatArray


def find_mu(
        s: FloatArray,
        y: FloatArray,
        eta: float = 1,
        tol: float = 1e-8,
        max_iter: int = 1000,
) -> float:
    """Solve for the empirical-Bayes noise parameter used in initialization."""
    logger = logging.getLogger(__name__)
    e = s ** 2
    z = y ** 2
    TM = z.size
    eta = eta * TM
    z2 = z.sum(axis=1)
    mu = 0
    diff = []

    logger.info('please wait: calculating mu...')
    for _ in range(max_iter):
        temp = 1 + mu * e
        fmu = z2 / (temp ** 2)
        f = fmu.sum() - eta
        dfmu = (-2) * fmu * e / temp
        diff.append(f / dfmu.sum())
        if (mu == 0 and f < 0) or abs(diff[-1] / diff[0]) < tol:
            logger.info(f"thanks for waiting, (mu: {mu}) calculation complete after iteration # {len(diff)} with relative error {diff[-1] / diff[0]}")
            return mu
        mu -= diff[-1]

    logger.info(f"maximum iteration {max_iter} reached, consider more iterations for convergence!")
    return mu


@dataclass(eq=False, repr=False)
class MNEInitializer:
    """MNE-style Gamma and data-covariance initializer for one lead field.

    The depth weighting and the SVD of the depth-weighted lead field depend only
    on the lead field, so one instance initializes any number of datasets --
    cross-validation folds, regularization candidates -- without repeating them.

    Parameters
    ----------
    lead_field
        Forward solution, shape ``(n_sensors, n_sources)``.
    use_depth_prior
        Weight sources by depth, compensating for the bias towards superficial
        sources.
    exp
        Exponent of the depth weighting.
    """

    lead_field: FloatArray
    use_depth_prior: bool = True
    exp: float = 0.8
    #: Depth weight of each source.
    w: FloatArray = field(init=False)
    #: SVD of the depth-weighted lead field.
    u: FloatArray = field(init=False)
    s: FloatArray = field(init=False)
    vh: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        l = self.lead_field
        if self.use_depth_prior:
            dw = 1.0 / (l ** 2).sum(axis=0)
            limit = dw.min() * 10.0
            self.w = np.minimum(dw / limit, 1) ** self.exp
        else:
            self.w = np.ones(l.shape[1])
        self.u, self.s, self.vh = linalg.svd(l * self.w[None, :], full_matrices=False)

    def __call__(self, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Initial source variances and sensor covariance for ``y``.

        Parameters
        ----------
        y
            Sensor measurement of one data segment, shape
            ``(n_sensors, n_times)``, whitened by the same filter as
            :attr:`lead_field`. ``Gamma`` averages over the ``n_times`` samples,
            so data that was normalized by ``sqrt(n_times)`` has to be rescaled
            before it is passed in.

        Returns
        -------
        Gamma
            Variance of each source, from the weighted least-squares estimate and
            its posterior covariance.
        data_cov
            Sensor covariance implied by ``Gamma``.
        """
        w, s, vh = self.w, self.s, self.vh
        yw = self.u.T @ y
        mu = find_mu(s, yw, eta=1)
        gamma = s / (s ** 2 + 1 / mu) if mu else 1 / s
        inv = ((w[:, None] * vh.T) * gamma[None, :]) @ yw
        # Gamma is the diagonal of ``inv @ inv.T / n_times + ecov``, with the posterior covariance
        #     ecov = mu * w[:, None] * (eye(n_sources) - vh.T @ ((gamma * s)[:, None] * vh)) * w[None, :]
        # Both terms are (n_sources, n_sources), so here their diagonals are computed directly rather than by forming the entire matrices: diag(A @ A.T) sums the squared rows of A, and the diagonal of ``vh.T @ diag(d) @ vh`` sums the squared rows of vh weighted by d.
        ecov_diagonal = mu * w ** 2 * (1 - ((gamma * s)[:, None] * vh ** 2).sum(0))
        Gamma = (inv ** 2).sum(1) / y.shape[1] + ecov_diagonal
        return Gamma, (self.lead_field * Gamma[None, :]) @ self.lead_field.T
