"""Champagne/Lasso NCRF solver.

This module contains the complete high-level implementation of the original
NCRF optimization algorithm. :class:`ChampLasso` is an immutable solver
configuration; :class:`_ChampLassoState` is private mutable state for one run.
"""
from __future__ import annotations

import copy
import logging
import time
import weakref
from dataclasses import dataclass, field, replace
from math import log10, sqrt
from multiprocessing import current_process
from numbers import Real
from typing import TYPE_CHECKING
from collections.abc import Sequence

from eelbrain import fmtxt
import numpy as np
from scipy import linalg
from scipy.signal import find_peaks
from tqdm import tqdm

from .._crossvalidation import crossvalidate
from .._fastac import Fasta
from .._linalg import _inv_sqrtm, _R_tol, compute_gamma
from .._penalties import g, g_group, proxg_group_opt, shrink
from .._repr import _count_repr, _theta_repr
from .._typing import FloatArray, GradientFunction, MuArg, ObjectiveFunction
from .base import Solver, SolverFit

if TYPE_CHECKING:
    from .._crossvalidation import CrossValidation, CVResult
    from .._data import RegressionData
    from .._forward import ForwardModel
    from .._model import NCRFEstimator

#: Per-iteration quantities :class:`ChampLasso` can record, in :class:`ChampLassoHistory`.
#: Names from this tuple are what :attr:`ChampLasso.store` selects from.
QUANTITIES = ('objective', 'relative_change', 'theta', 'gamma', 'sigma_b')

#: Champagne iterations for the warm covariance solve at ``theta = 0``, used
#: wherever a usable ``Sigma_b`` is needed before any FASTA step has run:
#: the ``mu == 0`` branch of ``run()`` and the grid calibration in ``gradient()``.
_N_ITERC_WARM = 30


@dataclass(repr=False)
class ChampLassoHistory:
    """Per-iteration quantities accumulated during fitting.

    :meth:`record` appends to a quantity's list only when ``store`` names it, so
    the amount of stored history can range from nothing to the full optimization
    trajectory.

    All lists are indexed by outer iteration. ``relative_change`` and ``theta`` are
    recorded for every iteration, whereas ``objective``, ``gamma`` and
    ``sigma_b`` come from the covariance update, which is skipped once the
    iterations converge; the last of those hence corresponds to
    ``theta[-2]`` when the solver stopped on the tolerance criterion.

    Attributes
    ----------
    store
        Which of :data:`QUANTITIES` to retain, taken from :attr:`ChampLasso.store`.
    objective
        Objective value after each covariance update.
    relative_change
        Relative change in ``theta`` after each outer iteration (the convergence
        criterion).
    theta, gamma, sigma_b
        Trajectories of the corresponding solver quantities. The last entry of
        each matches the corresponding attribute of the resulting
        :class:`ChampLassoFit`.
    """
    store: frozenset[str] = frozenset({'objective', 'relative_change'})
    objective: list[float] = field(default_factory=list)
    relative_change: list[float] = field(default_factory=list)
    theta: list[FloatArray] = field(default_factory=list)
    gamma: list = field(default_factory=list)
    sigma_b: list = field(default_factory=list)

    def __repr__(self) -> str:
        n_iterations = self.n_iterations
        stored = tuple(name for name in QUANTITIES if name in self.store)
        return f'<{type(self).__name__}: {_count_repr(n_iterations, "iteration")}, {stored=}>'

    @property
    def n_iterations(self) -> int:
        """Number of outer iterations that any quantity was recorded for."""
        return max((len(getattr(self, name)) for name in QUANTITIES), default=0)

    def record(self, **quantities: object) -> None:
        """Append the supplied quantities that :attr:`store` names.

        Parameters
        ----------
        quantities
            Values to record, keyed by their name in :data:`QUANTITIES`. A
            ``None`` value is skipped, so a caller that does not have every
            quantity at hand can pass what it has.

        Raises
        ------
        ValueError
            If a name is not one of :data:`QUANTITIES`, which would otherwise be
            silently dropped.
        """
        for name, value in quantities.items():
            if name not in QUANTITIES:
                raise ValueError(f"{name=}: not one of {QUANTITIES}")
            if name in self.store and value is not None:
                # deepcopy: gamma and sigma_b are lists of arrays the solver
                # keeps updating in place
                getattr(self, name).append(copy.deepcopy(value))


def _is_number(value: object) -> bool:
    """Whether ``value`` is a real number; ``bool`` is rejected."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _low_rank_sqrt(Cb: FloatArray, n_times: int) -> FloatArray:
    """Factor ``yhat`` with ``yhat @ yhat.T == Cb``, from the significant eigenvalues.

    Parameters
    ----------
    Cb
        Empirical data covariance, generally rank-deficient.
    n_times
        Number of samples the covariance was estimated from, which bounds its rank.
    """
    n_sensors = Cb.shape[0]
    lo = max(n_sensors - n_times, 0)
    e, v = linalg.eigh(Cb, subset_by_index=(lo, n_sensors - 1))
    indices = e > e[-1] * _R_tol
    return v[:, indices] * np.sqrt(e[indices])


def _whiten_by_sigma_b(
        sigma_b: FloatArray,
        *arrays: FloatArray,
) -> tuple[list[FloatArray], float]:
    """Whiten ``arrays`` by ``sigma_b``, falling back to its pseudo-inverse square root.

    Returns the whitened arrays along with ``log(det(sigma_b)) / 2``.
    """
    try:
        Lc = linalg.cholesky(sigma_b, lower=True)
        return [linalg.solve(Lc, array) for array in arrays], np.log(np.diag(Lc)).sum()
    except np.linalg.LinAlgError:
        # e holds the reciprocals of the significant eigenvalues, so
        # -log(e).sum() is the full (pseudo-)log-determinant
        Lc, e = _inv_sqrtm(sigma_b, return_eig=True)
        return [np.dot(Lc, array) for array in arrays], -np.log(e).sum() / 2


def _residual_factor(
        forward: ForwardModel,
        theta: FloatArray,
        meg: FloatArray,
        covariates: FloatArray,
) -> FloatArray:
    """Low-rank factor of one segment's residual covariance.

    The factor only ever enters through ``yhat @ yhat.T``, so any factor would
    do; the rank-revealing one has the fewest columns, which pays off in the
    per-source iterations of ``_solve()``.
    """
    y = meg - np.dot(np.dot(forward.whitened_lead_field, theta), covariates.T)
    return _low_rank_sqrt(np.dot(y, y.T), y.shape[1])


def _evaluate_objective(
        forward: ForwardModel,
        theta: FloatArray,
        Sigma_b: list,
        data: RegressionData,
) -> tuple[float, float]:
    """Evaluate the ChampLasso objective on whitened data.

    Returns the objective and its weighted-L2 term.
    """
    ll2 = 0
    logdet = 0
    for key, (meg, covariate) in enumerate(data):
        yhat = _residual_factor(forward, theta, meg, covariate)
        (y,), logdet_ = _whiten_by_sigma_b(Sigma_b[key], yhat)
        ll2 += 0.5 * (y ** 2).sum()
        logdet += logdet_
    return (ll2 + logdet) / len(data), ll2 / len(data)


#: Initialization seeds by dataset (see :func:`_mne_seeds`); entries die with their dataset.
_seed_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _mne_seeds(forward: ForwardModel, data: RegressionData) -> tuple[list, list[FloatArray]]:
    """Per-segment Champagne initialization seeds, cached by dataset.

    The seeds are a pure function of ``(forward, data)`` and are only ever read
    (every consumer copies before mutating), so they are shared: between the
    ``mu='auto'`` grid derivation and the final fit on the same data, and across
    the candidates a cross-validation worker scores on the same folds.
    """
    cached = _seed_cache.get(data)
    if cached is not None and cached[0] is forward:
        return cached[1], cached[2]
    init_gamma = []
    init_sigma_b = []
    for y, _ in data:
        t = y.shape[1]
        gamma, data_cov = forward.mne_initializer(y * (t ** 0.5))
        gamma = np.reshape(gamma, (-1, forward.dc))
        init_gamma.append([np.diag(g) for g in gamma])
        init_sigma_b.append(forward.whitened_noise_covariance + data_cov)
    _seed_cache[data] = (forward, init_gamma, init_sigma_b)
    return init_gamma, init_sigma_b


class _ChampLassoState:
    """Mutable state for one :meth:`ChampLasso.solve` call."""

    def __init__(
            self,
            solver: ChampLasso,
            forward: ForwardModel,
    ) -> None:
        # configuration (immutable)
        self.solver = solver
        self.forward = forward
        # initialization seeds (set by _initialize)
        self._init_gamma: list | None = None
        self._init_sigma_b: list[FloatArray] | None = None
        # working estimate (set during run())
        self.theta: FloatArray | None = None
        self.Gamma: list | None = None
        self.Sigma_b: list[FloatArray] | None = None

    def __repr__(self) -> str:
        solver = self.solver
        initialized = self.theta is not None
        details = f'{solver=!r}, {initialized=}'
        if initialized:
            details += f', {_theta_repr(self.theta)}'
        return f'<{type(self).__name__}: {details}>'

    def _initialize(self, data: RegressionData) -> None:
        """Seed the working state with a minimum-norm estimate."""
        # MNE-based seeds (re-read by _solve on every Champagne solve)
        self._init_gamma, self._init_sigma_b = _mne_seeds(self.forward, data)
        # Working estimate. _solve() replaces the per-segment entries wholesale
        # rather than writing into them, so the seeds can be shared.
        self.Gamma = list(self._init_gamma)
        self.Sigma_b = list(self._init_sigma_b)
        self.theta = np.zeros((self.forward.lead_field.shape[1], data.design.n_coefficients), dtype=np.float64)

    def _solve(
            self,
            data: RegressionData,
            theta: FloatArray,
            n_iterc: int | None = None,
    ) -> None:
        """Champagne steps implementation

        Parameters
        ----------
        data
            Whitened regression data to fit.
        theta
            Coefficients of the TRFs over the Gabor basis.

        Notes
        -----
        Implementation details can be found at:
        D. P. Wipf, J. P. Owen, H. T. Attias, K. Sekihara, and S. S. Nagarajan,
        “Robust Bayesian estimation of the location, orientation, and time course
        of multiple correlated neural sources using MEG,” NeuroImage, vol. 49,
        no. 1, pp. 641–655, 2010
        """
        logger = logging.getLogger('Champagne')
        if n_iterc is None:
            n_iterc = self.solver.n_iterc

        logger.debug('Champagne Iterations start:')
        logger.debug('trial \t time taken')
        for key, (meg, covariates) in enumerate(data):
            start = time.time()
            yhat = _residual_factor(self.forward, theta, meg, covariates)

            # compute_gamma() returns new objects, so the seed list only needs
            # to be shallow-copied; sigma_b is accumulated in place
            gamma = list(self._init_gamma[key])
            sigma_b = self._init_sigma_b[key].copy()

            # champagne iterations
            for _ in range(n_iterc):
                # pre-compute some useful matrices
                (lhat, ytilde), _ = _whiten_by_sigma_b(sigma_b, self.forward.whitened_lead_field, yhat)
                # sigma_b for the next iteration, accumulated over the sources
                sigma_b[:] = self.forward.whitened_noise_covariance[:]
                self._update_gamma(gamma, lhat, ytilde, sigma_b)

            self.Gamma[key] = gamma
            self.Sigma_b[key] = sigma_b
            end = time.time()
            logger.debug(f'{key} \t {end - start}')

    def _update_gamma(
            self,
            gamma: list,
            lhat: FloatArray,
            ytilde: FloatArray,
            sigma_b: FloatArray,
    ) -> None:
        """One Champagne sweep over the sources.

        Parameters
        ----------
        gamma
            Source covariances, updated in place.
        lhat, ytilde
            Lead field and data, whitened by the current ``sigma_b``.
        sigma_b
            Data covariance for the next iteration, accumulated in place; the
            caller seeds it with the noise covariance.
        """
        dc = self.forward.dc
        lead_field = self.forward.whitened_lead_field
        for i in range(len(self.forward.source)):
            block = self.forward.source_block(i)
            if dc > 1:
                x = np.dot(gamma[i], np.dot(lhat[:, block].T, ytilde))  # Xi
                z = np.dot(lhat[:, block].T, lhat[:, block])  # Zi
            else:
                x = gamma[i] * lhat[:, i].T.dot(ytilde)  # Xi
                z = (lhat[:, i] ** 2).sum()  # Zi
            gamma[i] = compute_gamma(z, x, dc)  # Ti
            sigma_b += np.dot(lead_field[:, block], np.dot(gamma[i], lead_field[:, block].T))

    def run(
            self,
            data: RegressionData,
            history: ChampLassoHistory,
            verbose: bool = False,
    ) -> None:
        """Run the alternating FASTA/Champagne optimization.

        Leaves ``theta``, ``Gamma`` and ``Sigma_b`` populated and records the
        requested per-iteration quantities into ``history``.
        """
        logger = logging.getLogger(__name__)
        mu = float(self.solver.mu)
        self._initialize(data)
        if mu == 0.0:
            self._solve(data, self.theta, n_iterc=_N_ITERC_WARM)

        if self.forward.space:
            def g_funct(x): return g_group(x, mu)
            def prox_g(x, t): return proxg_group_opt(x, mu * t)
        else:
            def g_funct(x): return g(x, mu)
            def prox_g(x, t): return shrink(x, mu * t)

        theta = self.theta
        myname = current_process().name

        if verbose:
            iter_o = tqdm(range(self.solver.n_iter))
        else:
            iter_o = range(self.solver.n_iter)

        # The objective is expensive to evaluate, so skip it when it would only
        # be discarded (e.g. cross-validation folds, which use store=())
        debug = logger.isEnabledFor(logging.DEBUG)
        logger.debug('process:iteration \t objective value \t %% change')
        for i in iter_o:
            funct, grad_funct = self._construct_f(data)
            if debug:
                logger.debug(f"Before FASTA:{funct(self.theta)}")
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.solver.n_iterf)
            Theta.learn(theta)

            change = self._relative_change(theta, Theta.coefs_)
            theta = Theta.coefs_
            self.theta = theta
            history.record(relative_change=change, theta=theta)
            if debug:
                logger.debug(f"After FASTA: {funct(self.theta)}")

            if change < self.solver.tol:
                break

            self._solve(data, theta)
            if debug or 'objective' in history.store:
                objective, _ = _evaluate_objective(self.forward, self.theta, self.Sigma_b, data)
                history.record(objective=objective)
                logger.debug(f'{myname}:{i} \t {objective} \t {change * 100}')
            history.record(gamma=self.Gamma, sigma_b=self.Sigma_b)

    def _construct_f(self, data: RegressionData) -> tuple[ObjectiveFunction, GradientFunction]:
        """Build the smooth objective and gradient passed to FASTA.

        Parameters
        ----------
        data
            Prepared regression data.
        """
        leadfields = []
        bEs = []
        bbts = []
        for i in range(len(data)):
            Linv = _inv_sqrtm(self.Sigma_b[i])
            leadfields.append(np.dot(Linv, self.forward.whitened_lead_field))
            bEs.append(np.dot(Linv, data.bE[i]))
            bbts.append(np.trace(np.dot(Linv, np.dot(Linv, data.bbt[i]).T)))

        def f(L, x, bbt, bE, EtE):
            Lx = np.dot(L, x)
            y = bbt - 2 * np.sum(bE * Lx) + np.sum(Lx * np.dot(Lx, EtE))
            return 0.5 * y

        def gradf(L, x, bE, EtE):
            y = bE - np.dot(np.dot(L, x), EtE)
            return -np.dot(L.T, y)

        def funct(x):
            fval = 0.0
            for i in range(len(data)):
                fval = fval + f(leadfields[i], x, bbts[i], bEs[i], data.EtE[i])
            return fval

        def grad_funct(x):
            grad = gradf(leadfields[0], x, bEs[0], data.EtE[0]).astype(np.float64)
            for i in range(1, len(data)):
                grad += gradf(leadfields[i], x, bEs[i], data.EtE[i])
            return grad

        return funct, grad_funct

    def gradient(self, data: RegressionData) -> FloatArray:
        """Per-source gradient magnitude of the data-fit term at the zero estimate.

        Runs an unregularized warm covariance solve and returns the magnitude of
        the smooth objective's gradient at ``theta = 0``, used to calibrate the
        automatic regularization grid. Independent of ``mu``.
        """
        self._initialize(data)
        self._solve(data, self.theta, n_iterc=_N_ITERC_WARM)
        _, grad_funct = self._construct_f(data)
        x = grad_funct(self.theta)
        if self.forward.space:
            x = x.reshape(-1, self.forward.dc, x.shape[1])
            return np.linalg.norm(x, axis=1)
        return np.abs(x)

    @staticmethod
    def _relative_change(theta0: FloatArray, theta1: FloatArray) -> float:
        diff = theta1 - theta0
        num = diff ** 2
        den = theta0 ** 2
        if den.sum() <= 0:
            return np.inf
        else:
            return sqrt(num.sum() / den.sum())


@dataclass(frozen=True, repr=False)
class ChampLassoFit(SolverFit):
    """Fitted state produced by :class:`ChampLasso`."""

    history: ChampLassoHistory
    gamma: list
    sigma_b: list[FloatArray]

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {_theta_repr(self.theta)}, {_count_repr(self.history.n_iterations, 'iteration')}, {_count_repr(len(self.sigma_b), 'segment')}>"

    def score(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> dict[str, float]:
        """Likelihood objective on whitened ``data`` and its weighted-L2 term."""
        cross_fit, weighted_l2_error = _evaluate_objective(forward, self.theta, self.sigma_b, data)
        return {'cross_fit': cross_fit, 'weighted_l2_error': weighted_l2_error}


def select_by_criterion(cv_results: Sequence[CVResult], criterion: str = 'cross-fit') -> ChampLasso:
    """Pick the best solver from cross-validation results by the given criterion.

    Parameters
    ----------
    cv_results
        Results to choose from.
    criterion
        Criterion for best fit. Possible values:

        - ``'cross-fit'``: The smallest cross-fit value (default)
        - ``'l2'``: The smallest l2 error
        - ``'l2/mu'``: The local minimum in the l2 error with the largest mu, i.e.
          the most regularized one (and hence the smallest TRF); falls back to the
          smallest l2 error when the l2 error has no local minimum
    """
    if criterion == 'cross-fit':
        return min(cv_results, key=lambda result: result.scores['cross_fit']).solver
    elif criterion == 'l2':
        return min(cv_results, key=lambda result: result.scores['l2_error']).solver
    elif criterion == 'l2/mu':
        results = sorted(cv_results, key=lambda result: result.solver.mu)
        peaks, _ = find_peaks([-result.scores['l2_error'] for result in results])  # find local minima
        if len(peaks) > 0:
            # higher mu -> smaller trf
            return max((results[peak] for peak in peaks), key=lambda result: result.solver.mu).solver
        return min(results, key=lambda result: result.scores['l2_error']).solver
    else:
        raise ValueError(f'{criterion=}')


def _select_es_solver(cv_results: Sequence[CVResult], minimum_mu: float) -> ChampLasso | None:
    """Return the solver at the first ES local minimum above ``minimum_mu``."""
    results = sorted(cv_results, key=lambda result: result.solver.mu)
    for i, result in enumerate(results[:-1]):
        if result.solver.mu < minimum_mu:
            continue
        if result.scores['estimation_stability'] < results[i + 1].scores['estimation_stability']:
            return result.solver
    return None


@dataclass(frozen=True)
class ChampLasso(Solver):
    """Alternating FASTA/Champagne solver for NCRF estimation.

    Parameters
    ----------
    mu
        Regularizer parameter. A number fits one model, a sequence selects among
        an explicit grid with cross-validation, and ``'auto'`` derives and
        cross-validates a grid from the data (default).
    n_iter
        Number of outer iterations of the algorithm.
    n_iterc
        Number of Champagne iterations within each outer iteration.
    n_iterf
        Number of FASTA iterations within each outer iteration.
    tol
        Tolerance factor deciding stopping criterion for the overall algorithm.
        Iteration stops when ``norm(trf_new - trf_old)/norm(trf_old) < tol``.
    use_es
        Refine the cross-validated ``mu`` with the estimation stability (ES) criterion
        :cite:`limEstimationStabilityCrossValidation2016` (default ``False``): among
        the candidates with ``mu`` equal or larger to the cross-fit winner, take the
        first local minimum of the ``estimation_stability`` score.
    store
        Which per-iteration quantities to keep in :attr:`ChampLassoFit.history`;
        any of ``'objective'``, ``'relative_change'``, ``'theta'``, ``'gamma'`` or
        ``'sigma_b'``. The two scalars (``'objective'`` and ``'relative_change'``, the
        default) are cheap; the other three retain the full trajectory and are
        correspondingly large.
    """

    mu: MuArg = 'auto'
    n_iter: int = 30
    n_iterc: int = 10
    n_iterf: int = 100
    tol: float = 1e-5
    use_es: bool = False
    store: Sequence[str] = ('objective', 'relative_change')

    def __post_init__(self) -> None:
        # a bare string would be iterated character by character
        if isinstance(self.store, str) or not frozenset(self.store).issubset(QUANTITIES):
            raise ValueError(f"store={self.store!r}: expected a sequence with any of {QUANTITIES}")

    def without_history(self) -> ChampLasso:
        """Disable all per-iteration storage for cross-validation folds."""
        return replace(self, store=())

    def search(
            self,
            estimator: NCRFEstimator,
            data: RegressionData,
            cv: CrossValidation,
    ) -> tuple[ChampLasso, list[CVResult]]:
        """Resolve ``mu``, and cross-validate unless it is a fixed number."""
        candidates = self.candidates(estimator.forward, data)
        if _is_number(self.mu):
            return candidates[0], []
        cv_results = crossvalidate(estimator, data, candidates, cv)
        # Extend before selecting, so that the estimation-stability criterion is
        # applied to the complete cross-fit search range
        extension = self._extend_grid(cv_results)
        if extension:
            cv_results.extend(crossvalidate(estimator, data, extension, cv))
        return self._select(cv_results), cv_results

    def _select(self, cv_results: Sequence[CVResult]) -> ChampLasso:
        """Select ``mu`` by cross-fit, optionally refined by estimation stability."""
        logger = logging.getLogger(__name__)
        solver = select_by_criterion(cv_results, 'cross-fit')
        if not self.use_es:
            return solver

        if solver.mu == max(result.solver.mu for result in cv_results):
            logger.info(f'\nCVmu is {solver.mu}: could not find mu based on estimation stability criterion\nContinuing with cross-validation only.')
            return solver
        es_solver = _select_es_solver(cv_results, solver.mu)
        if es_solver is None:
            logger.warning('\nNo ES minima found: could not find mu based on estimation stability criterion.\nContinuing with cross-validation only.')
            return solver
        return es_solver

    def _extend_grid(
            self,
            cv_results: Sequence[CVResult],
    ) -> tuple[ChampLasso, ...]:
        """Return one additional decade when the cross-fit winner is on a grid boundary."""
        logger = logging.getLogger(__name__)
        best = select_by_criterion(cv_results, 'cross-fit')
        mus = [result.solver.mu for result in cv_results]
        if best.mu == min(mus):
            if best.mu == 0.0:
                return ()  # cannot extend below the unregularized fit
            new_mus = np.logspace(np.log10(best.mu) - 1, np.log10(best.mu), 4)[:-1]
            direction = 'left'
        elif best.mu == max(mus):
            new_mus = np.logspace(np.log10(best.mu), np.log10(best.mu) + 1, 4)[1:]
            direction = 'right'
        else:
            return ()
        logger.info(f'Best cross-fit mu is {best.mu}: extending range of mu towards the {direction}')
        return tuple(replace(best, mu=float(mu)) for mu in new_mus)

    def cv_table(self, cv_results: Sequence[CVResult]) -> fmtxt.Table:
        """Summarize cross-validation scores by ``mu``.

        Call this on the solver that was selected by :meth:`search`;
        the table marks it among the candidates it was chosen from.
        The table also warns when its ``mu`` is at the bottom of the grid it was chosen from.

        Parameters
        ----------
        cv_results
            The results the selection was made from.
        """
        results = sorted(cv_results, key=lambda result: result.solver.mu)
        best_mu = {criterion: select_by_criterion(cv_results, criterion).mu for criterion in ('cross-fit', 'l2/mu')}

        table = fmtxt.Table('lllll')
        table.cells('mu', 'cross-fit', 'l2-error', 'weighted l2-error', 'ES metric')
        table.midrule()
        fmt = '%.5f'
        for result in results:
            text = fmtxt.stat(result.solver.mu, fmt=fmt)
            if result.solver == self:
                text += '*'
            table.cell(text)
            table.cell(fmtxt.stat(result.scores['cross_fit'], fmt, 1 if result.solver.mu == best_mu['cross-fit'] else 0, 1))
            table.cell(fmtxt.stat(result.scores['l2_error'], fmt, 1 if result.solver.mu == best_mu['l2/mu'] else 0, 1))
            table.cell(fmtxt.stat(result.scores['weighted_l2_error'], fmt=fmt))
            table.cell(fmtxt.stat(result.scores['estimation_stability'], fmt=fmt))
        if self.mu == min(result.solver.mu for result in results):
            table.caption("Warnings: Best mu is smallest mu")
        return table

    def candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[ChampLasso, ...]:
        """Resolve ``mu`` into fixed solver configurations."""
        mu = self.mu
        if isinstance(mu, str):
            if mu != 'auto':
                raise ValueError(f"{mu=}: expected a number, a sequence of numbers, or 'auto'")
            return self.auto_candidates(forward, data)
        if _is_number(mu):
            values = (mu,)
        else:
            try:
                values = tuple(mu)
            except TypeError:
                raise TypeError(f"{mu=}: expected a number, a sequence of numbers, or 'auto'") from None
            if not values:
                raise ValueError(f"{mu=}: grid must contain at least one value")
            if not all(_is_number(value) for value in values):
                raise TypeError(f"{mu=}: all grid values must be numbers")
        if any(value < 0 for value in values):
            raise ValueError(f"{mu=}: mu must be non-negative")
        return tuple(replace(self, mu=float(value)) for value in values)

    def solve(
            self,
            forward: ForwardModel,
            data: RegressionData,
            *,
            verbose: bool = False,
    ) -> ChampLassoFit:
        """Estimate NCRF weights for one prepared, whitened dataset."""
        if not _is_number(self.mu):
            raise ValueError("ChampLasso.solve() requires a fixed numeric mu; use NCRFEstimator.fit() to resolve a grid or mu='auto'")
        history = ChampLassoHistory(frozenset(self.store))
        state = _ChampLassoState(self, forward)
        state.run(data, history, verbose)
        return ChampLassoFit(
            theta=state.theta,
            history=history,
            gamma=state.Gamma,
            sigma_b=state.Sigma_b,
        )

    def auto_candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[ChampLasso, ...]:
        """Derive the standard seven-value ``mu`` grid from the data."""
        state = _ChampLassoState(self, forward)
        gradient = state.gradient(data)
        hi = log10(np.percentile(gradient, 99.0))
        mus = np.logspace(hi - 2, hi, 7)
        return tuple(replace(self, mu=float(mu)) for mu in mus)
