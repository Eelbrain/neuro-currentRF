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
from numbers import Integral
from typing import TYPE_CHECKING
from collections.abc import Sequence

from eelbrain import Split, Splits, split_data
from eelbrain.trf import init_worker
from eelbrain._config import CONFIG
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
    splits
        The cross-validation folds. An integer is the number of folds: the
        last ``splits`` windows of every segment (see ``validate``) are held
        out one at a time for validation, and the candidates are fit to the
        data preceding the window (``scheme='walk-forward'`` with
        ``split='within-case'`` and ``partitions=splits + 1`` in
        :func:`eelbrain.split_data`). The windows are cut from the samples
        whose lag window lies inside the stimulus, i.e., the samples the
        dataset covers. Alternatively, an
        :class:`eelbrain.Splits` object made for the data's segments, e.g.
        with :func:`eelbrain.split_data` and one of its other partitioning
        schemes; its validation splits are the folds. Splits with a test set
        are not supported yet. In either case, the training data of each fold
        excludes the samples within one TRF length (minus one sample) of the
        held-out data, so that the lag windows of the training and held-out
        samples never share a predictor sample.
    n_workers
        Number of worker processes, or ``None`` to derive it from Eelbrain's
        ``n_workers`` setting (see :func:`eelbrain.configure`). Set to ``0`` to
        run without :mod:`multiprocessing`, for debugging.
    validate
        Length of each validation window as proportion of the samples the
        dataset covers in each segment (default 0.05, i.e., 5%). Only applies
        when ``splits`` is the number of folds; a :class:`eelbrain.Splits` object defines its own
        validation sets.
    """

    splits: int | Splits = 3
    n_workers: int | None = None
    validate: float = 0.05

    def __post_init__(self) -> None:
        splits = self.splits
        if isinstance(splits, Splits):
            if splits.n_test:
                raise NotImplementedError(f"{splits}: cross-validation with a test set is not implemented; create the splits with test=0")
            elif not splits.n_validate:
                raise ValueError(f"{splits}: no validation set to score candidates on; create the splits with validate=1")
        elif isinstance(splits, Integral) and not isinstance(splits, bool):
            if splits < 1:
                raise ValueError(f"{splits=}: need at least one cross-validation fold")
        else:
            raise TypeError(f"{splits=}: need the number of partitions or an eelbrain Splits object")
        if not 0 < self.validate < 1:
            raise ValueError(f"validate={self.validate!r}: need a proportion of the segment, between 0 and 1")

    def worker_processes(self) -> int:
        """Number of worker processes, resolving ``n_workers=None`` from Eelbrain's setting."""
        if self.n_workers is not None:
            return self.n_workers
        # Eelbrain's setting is cpu_count() by default, and 0 when multiprocessing
        # is disabled. Use a fraction of the CPUs to avoid oversubscribing them,
        # since each worker's BLAS already runs its own thread pool
        return ceil(CONFIG['n_workers'] / 8)


def _training_ranges(
        split: Split,
        segments: np.ndarray,
        gap: int,
) -> np.ndarray:
    """``split.train`` without the ``gap`` samples on either side of each held-out range (within its segment).

    Parameters
    ----------
    split
        Split whose training ranges to trim.
    segments
        ``(n_segments, 2)`` array of the data segments on the concatenated time axis.
    gap
        Number of training samples to drop next to the held-out data.
    """
    train = np.zeros(segments[-1, 1], bool)
    for start, stop in split.train:
        train[start:stop] = True
    for start, stop in split.reserved:
        # held-out ranges never span a segment boundary
        seg_start, seg_stop = segments[np.searchsorted(segments[:, 1], start, 'right')]
        train[max(seg_start, start - gap):min(seg_stop, stop + gap)] = False
    # ranges that are adjacent across a segment boundary are merged here, and separated again by RegressionData.subset()
    return np.flatnonzero(np.diff(train, prepend=False, append=False)).reshape((-1, 2))


def _describe_blocks(blocks: Sequence[tuple[int, int, int]], max_items: int = 3) -> str:
    """Segment-relative sample ranges of ``blocks``, for error messages."""
    items = [f"segment {i}: samples {start}-{stop}" for i, start, stop in blocks[:max_items]]
    if len(blocks) > max_items:
        items.append('...')
    return ', '.join(items)


def _fold_ranges(data: RegressionData, cv: CrossValidation) -> list[tuple[np.ndarray, np.ndarray]]:
    """Training and held-out ranges of each fold of ``cv`` for ``data``, on the concatenated time axis.

    The folds are those of ``cv.splits``, or the last ``cv.splits`` windows of
    the samples ``data`` covers in every segment, held out in turn. Training
    data is separated from the held-out data by a gap of one TRF length minus
    one sample, so that the lag windows of training and held-out samples never
    share a predictor sample.

    The folds are checked against the samples ``data`` covers here, in the parent
    process, since an error in a :class:`~multiprocessing.pool.Pool` initializer,
    where the fold datasets are built, hangs the pool.

    Raises
    ------
    ValueError
        If the splits were made for different segments, or if a fold has no
        training or no held-out samples with a complete lag window. In
        particular, held-out data of a :class:`eelbrain.Splits` object at the
        start or end of a segment can lie entirely in the samples whose lag
        window extends beyond the stimulus.
    """
    if isinstance(cv.splits, Splits):
        if not np.array_equal(cv.splits.segments, data.data.segments):
            raise ValueError(f"{cv.splits}: made for different data segments than {data}")
        splits = cv.splits
    else:
        # cut the windows from the samples the dataset covers, so that none falls in the cropped segment ends
        offsets = data.data.segments[:, 0]
        covered = np.array([(offsets[i] + start, offsets[i] + stop) for i, start, stop in data.blocks])
        splits = split_data(covered, tstep=data.data.time.tstep, partitions=cv.splits + 1, split='within-case', scheme='walk-forward', validate=cv.validate, test=0)
    gap = max(data.design.stop_samples) - 1 - min(data.design.start_samples)  # largest minus smallest lag
    folds = []
    for i_fold, split in enumerate(splits.splits):
        train_ranges = _training_ranges(split, data.data.segments, gap)
        if not data._blocks_in(split.reserved):
            held_out = data.data.segment_slices(split.reserved)
            raise ValueError(f"cross-validation fold {i_fold}: none of its held-out samples ({_describe_blocks(held_out)}) has a complete lag window; the dataset only covers the samples whose lag window lies inside the stimulus ({_describe_blocks(data.blocks)}). Use longer segments, a TRF with tstart and tstop closer to 0, or Splits that hold out other samples")
        elif not data._blocks_in(train_ranges):
            raise ValueError(f"cross-validation fold {i_fold}: no training data remains after removing {gap} samples next to the held-out data; use fewer partitions, a shorter TRF, or more data")
        folds.append((train_ranges, split.reserved))
    return folds


def _make_folds(data: RegressionData, fold_ranges: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[tuple[RegressionData, RegressionData]]:
    """Train/validation fold datasets, shared by every candidate scored on ``data`` (see :func:`_fold_ranges`)."""
    return [(data.subset(train), data.subset(held_out)) for train, held_out in fold_ranges]


def _initialize_worker(data: RegressionData, fold_ranges: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Store the shared whitened data and the fold datasets once per multiprocessing worker."""
    global _worker_context
    init_worker()
    _worker_context = data, _make_folds(data, fold_ranges)


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
    n_processes = cv.worker_processes()
    fold_ranges = _fold_ranges(data, cv)
    results = []
    with tqdm(total=len(candidates), desc="Crossvalidation", unit='candidate', unit_scale=True) as prog:
        if n_processes == 0:
            folds = _make_folds(data, fold_ranges)
            for candidate in candidates:
                results.append(_score_candidate(data, folds, candidate))
                prog.update()
        else:
            with Pool(
                    processes=n_processes,
                    initializer=_initialize_worker,
                    initargs=(data, fold_ranges),
            ) as pool:
                for result in pool.imap_unordered(_score_worker, candidates):
                    results.append(result)
                    prog.update()

    return results
