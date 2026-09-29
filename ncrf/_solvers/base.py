"""The solver contract.

A solver is immutable configuration that estimates NCRF weights (:meth:`Solver.solve`).
A solver that has more than one configuration to choose from selects one in
:meth:`Solver.search`, calling ``crossvalidate`` as often as it needs to. That
one hook owns the whole search, so a solver can score a fixed grid, extend it,
or refine it in several passes without the estimator knowing anything about it.

:meth:`SolverFit.score` contributes solver-specific scores to compare
configurations by. Every hook has a working default, so a new solver only needs
:meth:`solve`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from collections.abc import Sequence

from eelbrain import fmtxt

from .._repr import _theta_repr
from .._typing import FloatArray

if TYPE_CHECKING:
    from .._crossvalidation import CrossValidation, CVResult
    from .._data import RegressionData
    from .._forward import ForwardModel


@dataclass(frozen=True, repr=False)
class SolverFit:
    """Fitted state from one solver execution.

    The generic fit contains the coefficient matrix consumed by
    :class:`~ncrf.NCRF`. Concrete solvers can add optimizer-specific state,
    diagnostics, and scores without coupling those details to the predictive
    model.

    Parameters
    ----------
    theta
        Fitted source-space coefficients over the regression design basis.
    """

    theta: FloatArray

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {_theta_repr(self.theta)}>'

    def score(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> dict[str, float]:
        """Solver-specific scores for ``data``, e.g. the training objective.

        Merged with the solver-independent model metrics wherever a fit is
        scored: on the training data in :attr:`NCRFFit.scores`, and per fold
        in ``CVResult.scores``. Keys that collide with a metric name are
        rejected by ``merge_scores``.
        """
        return {}


class Solver(ABC):
    """Configuration contract for an algorithm that estimates NCRF weights.

    :class:`~ncrf.NCRFEstimator` asks a solver to select a fixed configuration
    (:meth:`search`) and then calls :meth:`solve` for the final fit.
    """

    @abstractmethod
    def solve(
            self,
            forward: ForwardModel,
            data: RegressionData,
            *,
            verbose: bool = False,
    ) -> SolverFit:
        """Estimate source-space NCRF weights for prepared, whitened data."""

    def search(
            self,
            data: RegressionData,
            cv: CrossValidation,
    ) -> tuple[Solver, list[CVResult]]:
        """Select the fixed configuration to fit on all of the data.

        The default is a solver that is already fixed, and hence needs no
        cross-validation.

        Parameters
        ----------
        data
            Prepared, whitened data the fit will use; its
            :attr:`~ncrf.RegressionData.forward` is the forward model.
        cv
            Cross-validation configuration to score candidates with. Pass it to
            :func:`ncrf.crossvalidate` as often as the search needs;
            each call fits every candidate on every fold.

        Returns
        -------
        solver
            The configuration to fit, which has to be fixed enough for
            :meth:`solve`.
        cv_results
            Every result obtained from cross-validation, empty when the search
            did not cross-validate.
        """
        return self, []

    def without_history(self) -> Solver:
        """Return this configuration with per-iteration storage disabled.

        Used for cross-validation folds, whose history is discarded.
        """
        return self

    def cv_table(self, cv_results: Sequence[CVResult]) -> fmtxt.Table:
        """Summarize cross-validation scores in a table.

        Call this on the solver that was selected by :meth:`search`;
        the table marks it among the candidates it was chosen from.

        Parameters
        ----------
        cv_results
            The results the selection was made from.
        """
        keys = sorted(cv_results[0].scores)
        table = fmtxt.Table('l' * (len(keys) + 1))
        table.cells('solver', *keys)
        table.midrule()
        for result in cv_results:
            marker = '*' if result.solver == self else ''
            table.cell(f'{result.solver!r}{marker}')
            for key in keys:
                table.cell(fmtxt.stat(result.scores[key], fmt='%.5f'))
        return table
