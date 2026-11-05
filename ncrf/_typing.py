"""Shared type aliases for the NCRF package."""
from __future__ import annotations

from typing import Literal
from collections.abc import Callable, Sequence

from eelbrain import Datalist, NDVar
import mne
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]
TrialData = tuple[FloatArray, FloatArray]
NDVarArg = NDVar | Datalist | str
ObjectiveFunction = Callable[[FloatArray], float]
GradientFunction = Callable[[FloatArray], FloatArray]
MuArg = float | Sequence[float] | FloatArray | Literal["auto"]
NoiseArg = mne.Covariance | NDVar
Scale = Literal["l1", "l2", "spectral"]
ScaleArg = Scale | None
