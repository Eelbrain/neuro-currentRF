"""Public API for fitting and applying neuro-current response functions.

:func:`fit_ncrf` provides the complete convenience workflow. The component API
separates the :class:`RegressionData` dataset, the :class:`NCRFEstimator` fitting
orchestrator, :class:`ForwardModel` and :class:`TRFDesign` model components,
pluggable :class:`Solver` configurations, the reusable fitted :class:`NCRF`, and
the accompanying :class:`NCRFFit` report.
"""

from ._trf_design import TRFDesign
from ._data import RegressionData
from ._forward import ForwardModel
from ._crossvalidation import CrossValidation, CVResult, crossvalidate
from ._model import NCRFEstimator, NCRF, NCRFFit, fit_model
from ._metrics import explained_variance, l2_error
from ._solvers import ChampLasso, ChampLassoFit, Solver, SolverFit
from ._ncrf import fit_ncrf
