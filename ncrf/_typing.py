"""Shared type aliases for the NCRF package."""
from __future__ import annotations

from typing import Literal
from collections.abc import Callable, Sequence

from eelbrain import Categorial, NDVar, Scalar, Space
import mne
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]
TrialData = tuple[FloatArray, FloatArray]
ObjectiveFunction = Callable[[FloatArray], float]
GradientFunction = Callable[[FloatArray], FloatArray]
MuArg = float | Sequence[float] | FloatArray | Literal["auto"]
NoiseArg = mne.Covariance | NDVar
Scale = Literal["l1", "l2", "spectral"]
ScaleArg = Scale | None
StimDimensions = Categorial | Scalar | Space
