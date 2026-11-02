"""Solver-independent prediction metrics for fitted NCRF models.

Each metric maps observed and predicted whitened sensor data to a scalar. Metrics
are pure functions of the two sequences of per-segment arrays and know nothing
about the model that produced the predictions, so a caller can predict once and
evaluate several metrics on the same predictions; see :meth:`NCRF.evaluate`.
"""
from __future__ import annotations

from typing import TypeAlias
from collections.abc import Callable, Sequence

import numpy as np

from ._typing import FloatArray

#: A metric maps observed and predicted per-segment arrays to a scalar.
Metric: TypeAlias = Callable[[Sequence[FloatArray], Sequence[FloatArray]], float]


def merge_scores(*score_dicts: dict[str, float]) -> dict[str, float]:
    """Combine score mappings from different sources into one.

    Parameters
    ----------
    score_dicts
        Scores keyed by name, e.g. the model metrics and whatever
        :meth:`~ncrf.SolverFit.score` contributed.

    Raises
    ------
    ValueError
        If two sources provide the same key. Candidates are selected by score
        name, so a silent overwrite would change what cross-validation optimizes.
    """
    merged: dict[str, float] = {}
    for scores in score_dicts:
        duplicate = sorted(merged.keys() & scores.keys())
        if duplicate:
            raise ValueError(f"duplicate score {', '.join(duplicate)}: solver scores must not shadow the model metrics")
        merged.update(scores)
    return merged


def explained_variance(
        observed: Sequence[FloatArray],
        predicted: Sequence[FloatArray],
) -> float:
    """Explained variance in whitened sensor space: one minus the residual variance ratio, averaged over sensors and over blocks (weighted by their number of samples)."""
    residual_ratio = 0.0
    n_samples = 0
    for meg, prediction in zip(observed, predicted):
        residual = meg - prediction
        residual_ratio += np.nansum(np.var(residual, axis=1) / np.var(meg, axis=1)) / residual.shape[0] * meg.shape[1]
        n_samples += meg.shape[1]
    return 1 - residual_ratio / n_samples


def l2_error(
        observed: Sequence[FloatArray],
        predicted: Sequence[FloatArray],
) -> float:
    """Unweighted squared prediction error per sample in whitened sensor space (half the sum of squares over sensors, averaged over samples)."""
    error = 0.0
    n_samples = 0
    for meg, prediction in zip(observed, predicted):
        error += 0.5 * ((meg - prediction) ** 2).sum()
        n_samples += meg.shape[1]
    return error / n_samples
