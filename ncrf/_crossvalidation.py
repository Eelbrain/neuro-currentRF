"""Cross-validation machinery used by the NCRF estimator.

This module only scores fixed solver configurations: it partitions the data
with Eelbrain's cross-validation scheme, fits and evaluates each candidate on
every fold, and computes the estimation-stability metric across folds. Which
candidates to score, and which of them wins, is decided by
:meth:`~ncrf.Solver.search`.
"""

from __future__ import annotations

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)

import logging
from dataclasses import dataclass
from math import ceil
from multiprocessing import Pool
from typing import TYPE_CHECKING
from collections.abc import Sequence

from eelbrain.trf import Splits, init_worker, n_workers, split_data
import numpy as np
from tqdm import tqdm

from ._data import RegressionData
from ._metrics import merge_scores

if TYPE_CHECKING:
    from ._model import NCRF
    from ._solvers import Solver

_worker_context: tuple[RegressionData, list[tuple[RegressionData, RegressionData]]] | None = None


@dataclass(frozen=True)
class CrossValidation:
    """Configuration for selecting among solver candidates.

    Parameters
    ----------
    n_splits
        Number of cross-validation folds: the last ``n_splits`` windows of
        every segment, each 5% of the segment long, are held out one at a time
        for validation, and the candidates are fit to the data preceding the
        window (``split='ncrf'`` in :func:`eelbrain.trf.split_data`); this
        requires a single segment or segments of equal length.
    n_workers
        Number of worker processes, or ``None`` to derive it from Eelbrain's
        ``n_workers`` setting (see :func:`eelbrain.configure`). Set to ``0`` to
        run without :mod:`multiprocessing`, for debugging.
    """

    n_splits: int = 3
    n_workers: int | None = None

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(f"n_splits={self.n_splits!r}: need at least one cross-validation fold")


def _make_splits(data: RegressionData, n_splits: int) -> Splits:
    """The cross-validation folds for ``data``: the last ``n_splits`` windows of every segment, held out in turn."""
    return split_data(data.data.segments, n_splits, validate=1, test=0, split='ncrf')


def _make_folds(data: RegressionData, splits: Splits) -> list[tuple[RegressionData, RegressionData]]:
    """Train/validation fold datasets, shared by every candidate scored on ``data``."""
    return [(data.subset(split.train), data.subset(split.reserved)) for split in splits.splits]


def _initialize_worker(data: RegressionData, splits: Splits) -> None:
    """Store the shared whitened data and the fold datasets once per multiprocessing worker."""
    global _worker_context
    init_worker()
    _worker_context = data, _make_folds(data, splits)


def _score_worker(solver: Solver) -> CVResult:
    if _worker_context is None:
        raise RuntimeError("cross-validation worker was not initialized")
    data, folds = _worker_context
    return _score_candidate(data, folds, solver)


def compute_es_metric(models: Sequence[NCRF], data: RegressionData) -> float:
    """Compute the estimation-stability metric across cross-validation folds.

    Details can be found at:
    Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
    Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

    Parameters
    ----------
    models
        Fitted fold models from cross-validation.
    data
        Dataset used to compare their predictions.

    Returns
    -------
    float
        Estimation-stability score.
    """
    Y = np.array([np.concatenate([prediction.ravel() for prediction in model.predict(data, whitened=True)]) for model in models])
    Y_bar = Y.mean(axis=0)
    VarY = (((Y - Y_bar) ** 2).sum(axis=1)).mean()
    denominator = (Y_bar ** 2).sum()
    # NaN predictions (a diverged fit) are as unstable as it gets
    if denominator <= 0 or np.isnan(VarY):
        return np.inf
    return VarY / denominator


@dataclass(frozen=True)
class CVResult:
    """Cross-validation scores for one solver candidate.

    Parameters
    ----------
    solver
        Candidate that was evaluated.
    scores
        Mean held-out scores across folds. Always contains the solver-independent
        model metrics (``explained_variance``, ``l2_error``) and
        ``estimation_stability``; solvers add their own through
        :meth:`SolverFit.score`.
    """

    solver: Solver
    scores: dict[str, float]


def _score_candidate(
        data: RegressionData,
        folds: Sequence[tuple[RegressionData, RegressionData]],
        solver: Solver,
) -> CVResult:
    """Fit and score all cross-validation folds for one solver candidate.

    Each fold is fit through :func:`~ncrf.fit_model`, then scored on its held-out
    window with the model metrics plus whatever the solver's fit contributes.
    """
    from ._model import fit_model, _score_fit  # _model imports this module

    fold_solver = solver.without_history()
    models = []
    fold_scores = []
    for traindata, testdata in folds:
        model, solver_fit = fit_model(traindata, fold_solver)
        models.append(model)
        fold_scores.append(_score_fit(model, solver_fit, testdata))

    scores = {key: sum(fold[key] for fold in fold_scores) / len(fold_scores) for key in fold_scores[0]}
    estimation_stability = compute_es_metric(models, data)
    return CVResult(solver, merge_scores(scores, {'estimation_stability': estimation_stability}))


def crossvalidate(
        data: RegressionData,
        candidates: Sequence[Solver],
        cv: CrossValidation,
) -> list[CVResult]:
    """Perform cross-validation over a set of solver candidates.

    Each candidate is fit and scored on the same folds, and the resulting
    :class:`CVResult` objects are returned for the caller to compare. This is what
    :meth:`~ncrf.Solver.search` calls to score the configurations it chooses between.

    Parameters
    ----------
    data
        Whitened M/EEG data and the corresponding stimulus variables (see
        :meth:`RegressionData.whiten`); its forward model is the one the
        candidates are fit with. It is pickled to send it to worker processes.
    candidates
        Fixed solver configurations to compare.
    cv
        Folds and worker count to score the candidates on.

    Returns
    -------
    list
        Cross-validation results.
    """
    logging.getLogger(__name__).info('Crossvalidation initiated!')
    n_processes = cv.n_workers
    if n_processes is None:
        # Eelbrain's setting is cpu_count() by default, and 0 when multiprocessing
        # is disabled. Use a fraction of the CPUs to avoid oversubscribing them,
        # since each worker's BLAS already runs its own thread pool
        n_processes = ceil(n_workers() / 8)

    # Partition in the parent process: an error in a Pool initializer hangs the pool
    splits = _make_splits(data, cv.n_splits)
    results = []
    with tqdm(total=len(candidates), desc="Crossvalidation", unit='candidate', unit_scale=True) as prog:
        if n_processes == 0:
            folds = _make_folds(data, splits)
            for candidate in candidates:
                results.append(_score_candidate(data, folds, candidate))
                prog.update()
        else:
            with Pool(
                    processes=n_processes,
                    initializer=_initialize_worker,
                    initargs=(data, splits),
            ) as pool:
                for result in pool.imap_unordered(_score_worker, candidates):
                    results.append(result)
                    prog.update()

    return results
