"""Public package surface for the NCRF fitting pipeline.

The package is organized around a small top-level API: :func:`fit_ncrf`
coordinates input normalization and model fitting, while :class:`NCRF` and
:class:`RegressionData` expose the lower-level object-oriented workflow.
"""

from ._model import NCRF, RegressionData, gaussian_basis
from ._ncrf import fit_ncrf
