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

from importlib.metadata import version
from pathlib import Path

try:
    __version__ = version("ncrf")
except Exception:
    __version__ = "0.0.0"
# In a git checkout (e.g., editable install), derive the current version from git
if (Path(__file__).parent.parent / '.git').exists():
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
        del get_version
    except Exception:
        pass
del version, Path
