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
from ._model import NCRFEstimator, NCRF, NCRFExecutionContext, NCRFFit, fit_model
from ._metrics import explained_variance, l2_error
from ._solvers import ChampLasso, ChampLassoFit, Solver, SolverFit
from ._ncrf import fit_ncrf

from pathlib import Path

__version__ = None
if (Path(__file__).parents[1] / '.git').exists():
    # Git checkout (e.g., editable install): _version.py is not updated with new commits
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
        del get_version
    except Exception:
        pass
if __version__ is None:
    try:
        # Written by setuptools_scm when the package is built
        from ._version import __version__
    except ImportError:
        # Source tree that has never been built
        __version__ = '0.0.0'
del Path
