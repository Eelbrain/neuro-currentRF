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
    """Mean sensor-by-segment explained variance in whitened sensor space."""
    residual_ratio = 0.0
    for meg, prediction in zip(observed, predicted):
        residual = meg - prediction
        residual_ratio += np.nansum(np.var(residual, axis=1) / np.var(meg, axis=1)) / residual.shape[0]
    return 1 - residual_ratio / len(observed)


def l2_error(
        observed: Sequence[FloatArray],
        predicted: Sequence[FloatArray],
) -> float:
    """Mean unweighted squared prediction error in whitened sensor space."""
    error = 0.0
    for meg, prediction in zip(observed, predicted):
        error += 0.5 * ((meg - prediction) ** 2).sum()
    return error / len(observed)
