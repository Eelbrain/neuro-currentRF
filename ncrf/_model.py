"""Core data structures and optimization logic for NCRF estimation.

``RegressionData`` turns Eelbrain objects into normalized numeric arrays with a
stable internal layout, and :class:`NCRF` consumes that prepared data to run
the alternating Bayesian/source-estimation procedure.
"""
# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Christian Brodbeck <email:brodbecc@mcmaster.ca>
# License: BSD (3-clause)
from __future__ import annotations

import time
import copy
import collections
from dataclasses import dataclass, field
from functools import cached_property, wraps
from math import sqrt, log10
from multiprocessing import current_process
from operator import attrgetter
from typing import Any, Callable, Iterator, Literal, Sequence

from eelbrain import Categorial, NDVar, Scalar, Sensor, Space, UTS, fmtxt
import numpy as np
import numpy.typing as npt
from scipy import linalg
from scipy.signal import find_peaks
from tqdm import tqdm

from ._fastac import Fasta
from ._crossvalidation import CVResult, crossvalidate
from . import opt
from .dsyevh3C import compute_gamma_c

import logging


_R_tol = np.finfo(np.float64).eps * 1e2
FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]
TrialData = tuple[FloatArray, FloatArray]
ObjectiveFunction = Callable[[FloatArray], float]
GradientFunction = Callable[[FloatArray], FloatArray]
MuArg = float | Sequence[float] | Literal["auto"]
MusArg = Sequence[float] | Literal["auto"] | None
StimDimensions = Categorial | Scalar | Space


def gaussian_basis(
        n_atoms: int,
        lags: npt.ArrayLike,
        basis_std: float = 0.0085,
) -> FloatArray:
    """Construct Gabor basis for the TRFs.

    Parameters
    ----------
    n_atoms
        number of atoms
    lags
        One-dimensional lag times covered by the basis functions,
        shape ``(n_lags,)``.
    basis_std
        Standard deviation of each Gaussian atom, expressed in the same units
        as ``lags``.

    Returns
    -------
    ndarray
        Array whose columns contain the basis atoms. Shape ``(n_lags, n_basis)``, with
        ``n_basis = nlevel - 1``.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Using gaussian basis with {basis_std=}')
    lags = np.asarray(lags, dtype=np.float64)
    lag_start = lags[0]
    lag_stop = lags[-1]
    lag_step = (lag_stop - lag_start) / n_atoms
    centers = np.linspace(lag_start + lag_step, lag_stop - lag_step, num=n_atoms - 1)
    basis = np.exp(-((lags[:, None] - centers[None, :]) ** 2) / (2 * basis_std ** 2))
    return basis / basis.max()


def g(x: FloatArray, mu: float) -> float:
    """Vector l1-norm penalty."""
    return mu * np.sum(np.abs(x))


def proxg(x: FloatArray, mu: float, tau: float) -> FloatArray:
    """Proximal operator for the l1-norm penalty."""
    return shrink(x, mu * tau)


def shrink(x: FloatArray, mu: float) -> FloatArray:
    """Soft-thresholding operator."""
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def g_group(x: FloatArray, mu: float) -> float:
    r"""group (l12) norm penalty:

            gg(x) = \sum ||x_s_{i,t}||

    where s_{i,t} = {x_{j,t}: j = 1*dc:(i+1)*dc}, i \in {1,2,...,#sources}, t \in {1,2,...,M}
    """
    l = x.shape[1]
    x.shape = (-1, 3, l)
    val = mu * np.sqrt((x ** 2).sum(axis=1)).sum()
    x.shape = (-1, l)
    return val


def proxg_group_opt(z: FloatArray, mu: float) -> FloatArray:
    """proximal operator for gg(x):

            prox_{mu gg}(x) = min  gg(z) + 1/ (2 * mu) ||x-z|| ** 2
                    x_s = max(1 - mu/||z_s||, 0) z_s

    Note: It does update the supplied z. It is a wrapper for distributed Cython code.
    """
    # x = z.view()
    l = z.shape[1]
    z.shape = (-1, 3, l)
    opt.cproxg_group(z, mu, z)
    z.shape = (-1, l)
    return z


def covariate_from_stim(
        stims: Sequence[NDVar] | NDVar,
        Ms: Sequence[int] | npt.ArrayLike,
        starts: Sequence[int] | npt.ArrayLike,
) -> list[FloatArray]:
    """Form lagged covariate matrices from one or more stimulus NDVars.

    Parameters
    ----------
    stims
        Predictor variables. Each predictor must provide a ``time`` axis and may have
        at most one additional feature dimension before time.
    Ms
        Filter lengths, in samples, for each expanded predictor channel.
    starts
        Start offsets, in samples, for each expanded predictor channel.

    Returns
    -------
    list
        Covariate matrices, one per expanded predictor channel. Each matrix has
        one row per stimulus time sample; rows with incomplete stimulus history are
        zero-padded.
    """
    ws = []
    for stim in stims:
        if stim.ndim == 1:
            w = stim.get_data((np.newaxis, 'time'))
        else:
            dimnames = stim.get_dimnames(last='time')
            w = stim.get_data(dimnames)
        ws.append(w)
    ws = ws[0] if len(ws) == 1 else np.concatenate(ws, 0)
    assert len(ws) == len(Ms) == len(starts), f"Length of w ({len(ws)}), Ms ({len(Ms)}), and start ({len(starts)}) should be equal"

    n_times = ws.shape[1]
    Y = []
    for w, start, M in zip(ws, starts, Ms):
        X = np.zeros((n_times, M), dtype=w.dtype)
        for i in range(n_times):
            stop = i + 1
            start_i = max(0, stop - M)
            n = stop - start_i
            X[i, :n] = w[start_i:stop][::-1]
        if start != 0:
            # -ve tstart -> shift covariate matrix left
            # +ve tstart -> shift covariate matrix right
            X = np.roll(X, start, axis=0)
            if start < 0:
                X[start:] = 0
            else:
                X[:start] = 0
        Y.append(X)
    return Y


def _myinv(x: FloatArray) -> FloatArray:
    """Compute a tolerance-aware elementwise reciprocal."""
    x = np.real(np.array(x))
    tol = _R_tol * x.max()
    ind = (x > tol)
    y = np.zeros(x.shape)
    y[ind] = 1 / x[ind]
    return y


def _inv_sqrtm(
        m: FloatArray,
        return_eig: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    e, v = linalg.eigh(m)
    e = e.real
    tol = _R_tol * e.max()
    ind = (e > tol)
    y = np.zeros((e.shape[0], 1))
    y[ind, 0] = 1 / e[ind]
    if return_eig:
        return np.sqrt(y) * v.T.conj(), np.squeeze(y[ind])
    return np.sqrt(y) * v.T.conj()


def _compute_gamma_i(z: FloatArray, x: FloatArray) -> FloatArray:
    """Computes Gamma_i

    Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
           = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
           = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
           = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    Parameters
    ----------
    z
        Auxiliary matrix for one source block.
    x
        Auxiliary coefficients for the same source block.

    Returns
    -------
    ndarray
        Updated block covariance matrix.
    """
    [e, v] = linalg.eig(z)
    e = e.real
    e[e < 0] = 0
    temp = np.dot(x.T, v)
    temp = np.real(np.dot(temp.conj().T, temp))
    e = np.sqrt(e)
    [d, u] = linalg.eig((temp * e) * e[:, np.newaxis])
    d = d.real
    d[d < 0] = 0
    d = np.sqrt(d)
    temp = np.dot(v * _myinv(np.real(e)), u)
    return np.array(np.real(np.dot(temp * d, temp.conj().T)))


def _compute_gamma_ip(z: FloatArray, x: FloatArray, gamma: FloatArray) -> None:
    """Wrapper function of Cython function 'compute_gamma_c'

    Computes Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
                   = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
                   = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
                   = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    Parameters
    ----------
    z
        Auxiliary square matrix for one source block, usually of shape ``(dc, dc)``.
    x
        Auxiliary coefficients for the same source block.
    gamma
        Output array that is updated in place with the new block covariance.
    """
    assert x.shape[0] == 3
    a = np.dot(x, x.T)
    compute_gamma_c(z, a, gamma)
    return


@dataclass
class OptimizationTracker:
    """Records per-iteration snapshots during NCRF.fit()."""
    snapshots: list = field(default_factory=list)

    def record(self, iteration: int, objective: float, residual: float, theta: FloatArray):
        self.snapshots.append({
            'iteration': iteration,
            'objective': objective,
            'residual': residual,
            'theta': theta.copy(),
        })

    def summary(self):
        for s in self.snapshots:
            print(f"Iter {s['iteration']:3d}  obj={s['objective']:.6f}  residual={s['residual']:.2e}")


@dataclass(eq=False, repr=False)
class RegressionData:
    """Prepared dataset for NCRF fitting.

    Use :meth:`from_data` to construct a dataset from raw MEG and stimulus
    :class:`~eelbrain.NDVar` objects.

    Parameters
    ----------
    meg
        MEG signal arrays, one per segment, each shaped
        ``(n_sensors, n_times)``.
    covariates
        Basis-projected covariate matrices, one per segment, each shaped
        ``(n_times, n_basis_cols)``.
    norm_factor
        ``sqrt(n_times)`` of the first segment; used by :meth:`timeslice`
        to rescale sub-segments consistently.
    basis
        Gaussian basis matrices, one per predictor variable, each shaped
        ``(filter_length, n_basis)``.
    tstart
        TRF start time in seconds, one value per predictor.
    tstep
        Sample spacing in seconds, shared by all segments.
    tstop
        TRF stop time in seconds, one value per predictor.
    stim_is_single
        ``True`` when the original stimulus input contained a single
        predictor per segment rather than a list; controls whether
        :attr:`NCRF.h` returns a bare NDVar or a list.
    stim_dims
        Feature dimension for each predictor (``None`` for scalar predictors).
    stim_names
        Name of each predictor variable.
    baseline
        Per-predictor centering values subtracted before covariate
        construction, or ``None`` if no centering was applied.
    scaling
        Per-predictor scale factors applied after centering, or ``None``
        if no scaling was applied.
    stim_normalization
        Spectral norms of each predictor block before post-normalization,
        one inner list per segment.
    basis_std
        Standard deviation of the Gaussian basis functions in seconds.
    sensor_dim
        Sensor dimension shared by all MEG segments.
    is_whitened
        Whether ``meg`` has already been transformed by a whitening filter.
    """

    meg: list[FloatArray]  # (sensor, time)
    covariates: list[FloatArray]  # (time, covariate)
    norm_factor: float
    basis: list[FloatArray]  # (filter_time, covariate)
    tstart: list[float]
    tstep: float
    tstop: list[float]
    stim_is_single: bool
    stim_dims: list[StimDimensions | None]
    stim_names: list[str]
    baseline: Sequence[NDVar | float] | None
    scaling: Sequence[NDVar | float] | None
    stim_normalization: list[list[float]]  # (segment, expanded covariate)
    basis_std: float
    sensor_dim: Sensor
    is_whitened: bool = False

    def __post_init__(self) -> None:
        if len({m.shape[1] for m in self.meg}) > 1:
            raise NotImplementedError("Segments with unequal trial length")

    @classmethod
    def from_data(
            cls,
            meg: list[NDVar],
            stim: list[Sequence[NDVar]],
            tstart: float | Sequence[float],
            tstop: float | Sequence[float],
            nlevel: int = 1,
            baseline: Sequence[NDVar | float] | None = None,
            scaling: Sequence[NDVar | float] | None = None,
            stim_is_single: bool = False,
            basis_std: float = 0.0085,
            in_place: bool = False,
            post_normalize: bool = True,
            pad_stim: bool = False,
    ) -> RegressionData:
        """Construct a dataset from MEG and stimulus NDVars.

        Parameters
        ----------
        meg
            MEG segments, each an NDVar with ``sensor`` and ``time`` dimensions.
        stim
            Stimulus lists, one per segment; each inner list contains one NDVar per
            predictor. Each predictor may be 1-D over time or carry one feature
            dimension before time.
        tstart
            Start of the TRF in seconds. A scalar applies to all predictors; a
            sequence specifies one start time per predictor.
        tstop
            Stop of the TRF in seconds. A scalar applies to all predictors; a
            sequence specifies one stop time per predictor.
        nlevel
            Density of Gabor basis atoms. Bigger → less dense. ``nlevel > 2``
            should be used with caution.
        baseline
            Per-predictor means to subtract from ``stim`` before covariate
            construction.
        scaling
            Per-predictor scaling factors applied after baseline subtraction.
        stim_is_single
            Whether the original stimulus input was a single predictor per segment.
        basis_std
            Standard deviation of the Gaussian basis functions in seconds.
        in_place
            If ``False`` (default), copies of ``stim`` are made before applying
            baseline subtraction or scaling. Set to ``True`` to modify in place.
        post_normalize
            If ``True`` (default), equalize covariate scales across predictor
            blocks by dividing each block by its average spectral norm.
        pad_stim
            If ``False`` (default), keep only rows whose full lag window is inside
            the stimulus time axis. If ``True``, retain edge rows with zero-padded
            covariates.
        """
        if not meg:
            raise ValueError("meg is empty")
        elif len(meg) != len(stim):
            raise ValueError("meg and stim have different lengths")

        tstart = list(tstart) if isinstance(tstart, collections.abc.Sequence) else [tstart]
        tstop = list(tstop) if isinstance(tstop, collections.abc.Sequence) else [tstop]

        # State initialized from the first segment and compared to subsequent segments
        sensor_dim = None
        stim_dims = None
        stim_names = None
        tstep = None
        basis = None
        filter_length = None
        row_slice = None
        start_samples = None  # in-samples offsets, local to from_data

        meg_arrays: list[FloatArray] = []
        covariate_arrays: list[FloatArray] = []
        s_normalization = []
        trial_length = None
        norm_factor = None

        for i_segment, (m, ss) in enumerate(zip(meg, stim)):
            if in_place:
                ss = list(ss)
            else:
                ss = [s.copy() for s in ss]

            # Sensor dim
            if sensor_dim is None:
                sensor_dim = m.get_dim('sensor')
            elif m.get_dim('sensor') != sensor_dim:
                raise ValueError(f'{meg=}: combining data segments with different sensor configurations is not supported')

            # Time dim
            meg_time: UTS = m.get_dim('time')
            if tstep is None:
                tstep = meg_time.tstep
            elif meg_time.tstep != tstep:
                raise ValueError(f"{meg=}: segment {i_segment} time-step incompatible with first segment")
            if trial_length is None:
                trial_length = len(meg_time)
            elif len(meg_time) != trial_length:
                raise NotImplementedError(f"{meg=}: unequal trial length")

            # Determine stim feature dims for this segment
            cur_stim_dims = []
            for x in ss:
                if x.get_dim('time') != meg_time:
                    raise ValueError(f"segment {i_segment} stim {x!r}: time axis incompatible with meg")
                elif x.ndim == 1:
                    cur_stim_dims.append(None)
                elif x.ndim == 2:
                    dim, _ = x.get_dims((None, 'time'))
                    cur_stim_dims.append(dim)
                else:
                    raise ValueError(f"Segment {i_segment} stim {x!r}: more than 2 dimensions")

            if stim_dims is None:
                # Initialize time/basis parameters from the first segment
                stim_dims = cur_stim_dims
                stim_names = [x.name for x in ss]
                if len(tstart) == 1:
                    tstart = tstart * len(stim_dims)
                if len(tstop) == 1:
                    tstop = tstop * len(stim_dims)
                assert len(tstart) == len(stim_dims)
                assert len(tstop) == len(stim_dims)
                start_samples = [int(round(ts / tstep)) for ts in tstart]
                stop_samples = [int(round(te / tstep)) for te in tstop]
                filter_length = np.subtract(stop_samples, start_samples) + 1
                basis = []
                for ts, te, fl in zip(tstart, tstop, filter_length):
                    x = np.linspace(ts, te, fl)
                    basis.append(gaussian_basis(int(round((fl - 1) / nlevel)), x, basis_std))
                if not pad_stim:
                    # covariate_from_stim() fills the full MEG axis with
                    # zero-padded lag histories. ``row_slice`` keeps only samples
                    # whose complete lag window lies inside the stimulus.
                    drop_start = max(0, *stop_samples)
                    drop_stop = max(0, *(-s for s in start_samples))
                    if drop_start or drop_stop:
                        row_slice = slice(drop_start, -drop_stop if drop_stop else None)
            elif cur_stim_dims != stim_dims:
                raise ValueError(f"{stim=}: segment {i_segment} dimensions incompatible with first segment")

            # Apply stim normalization
            if baseline is not None:
                if len(baseline) != len(ss):
                    raise ValueError(f"baseline length {len(baseline)} != number of predictors {len(ss)}")
                for s, b in zip(ss, baseline):
                    s -= b
            if scaling is not None:
                if len(scaling) != len(ss):
                    raise ValueError(f"scaling length {len(scaling)} != number of predictors {len(ss)}")
                for s, sc in zip(ss, scaling):
                    s /= sc

            # Extract and normalize MEG array
            y = m.get_data(('sensor', 'time'))
            y_ = y.astype(np.float64, copy=False)
            y = y_ if (in_place or y_.base is None) else y_.copy()

            # Build basis-projected covariate matrix
            stim_lens = [len(d) if d else 1 for d in stim_dims]
            fl_rep = np.repeat(np.asanyarray(filter_length), stim_lens)
            st_rep = np.repeat(np.asanyarray(start_samples), stim_lens)
            raw_covs = covariate_from_stim(ss, fl_rep, st_rep)

            if row_slice is not None:
                y = y[:, row_slice]
                raw_covs = [x[row_slice] for x in raw_covs]
            if not y.shape[1]:
                raise ValueError(f"{meg=}: no samples remain after applying lag-validity crop")
            flat = np.var(y, axis=1) == 0
            if flat.any():
                raise ValueError(f"{meg=}: segment {i_segment} has flat channels ({', '.join(sensor_dim.names[flat])})")
            norm_factor = sqrt(y.shape[1])
            y /= norm_factor
            meg_arrays.append(y)

            i = 0
            covariates = []
            for d, b in zip(stim_dims, basis):
                l = len(d) if d else 1
                covariates.extend([np.dot(x, b) / sqrt(y.shape[1]) for x in raw_covs[i:i + l]])
                i += l
            s_normalization.append([linalg.norm(x, 2) for x in covariates])
            covariate_arrays.append(np.concatenate(covariates, axis=1).astype(np.float64))

        # Equalize covariate scales across predictor blocks
        if post_normalize:
            n_vars = sum(len(d) if d else 1 for d in stim_dims)
            if n_vars > 1:
                stim_lens = [len(d) if d else 1 for d in stim_dims]
                bl_lengths = np.repeat([b.shape[1] for b in basis], stim_lens)
                avg_norm = np.array(s_normalization).mean(axis=0)
                col = 0
                for bl, norm in zip(bl_lengths, avg_norm):
                    for cov in covariate_arrays:
                        cov[:, col:col + bl] /= norm
                    col += bl

        return cls(
            meg_arrays, covariate_arrays, norm_factor,
            basis=basis,
            tstart=tstart, tstep=tstep, tstop=tstop,
            stim_is_single=stim_is_single, stim_dims=stim_dims, stim_names=stim_names,
            baseline=baseline, scaling=scaling,
            stim_normalization=s_normalization, basis_std=basis_std,
            sensor_dim=sensor_dim,
        )

    def __iter__(self) -> Iterator[TrialData]:
        return zip(self.meg, self.covariates)

    def __len__(self) -> int:
        return len(self.meg)

    def __repr__(self) -> str:
        return 'Regression data'

    @cached_property
    def bbt(self) -> list[FloatArray]:
        """Per-segment ``B @ B.T`` matrices for stored MEG arrays."""
        return [np.dot(b, b.T) for b in self.meg]

    @cached_property
    def bE(self) -> list[FloatArray]:
        """Per-segment ``B @ E`` cross-product matrices."""
        return [np.dot(b, E) for b, E in zip(self.meg, self.covariates)]

    @cached_property
    def EtE(self) -> list[FloatArray]:
        """Per-segment ``E.T @ E`` covariate Gram matrices."""
        return [np.dot(E.T, E) for E in self.covariates]

    def whiten(self, whitening_filter: FloatArray) -> RegressionData:
        """Return a new dataset with MEG whitened.

        Parameters
        ----------
        whitening_filter
            Whitening matrix.

        Notes
        -----
        Uses shallow copies of unmodified data.

        Raises
        ------
        ValueError
            If the dataset is already whitened. Whitening twice is not equivalent
            to whitening once with the second filter (``W₂ @ W₁ @ meg ≠ W₂ @ meg``).
        """
        if self.is_whitened:
            raise ValueError("Dataset is already whitened; cannot whiten twice")
        meg = [np.dot(whitening_filter, m) for m in self.meg]
        return RegressionData(
            meg, self.covariates, self.norm_factor,
            basis=self.basis,
            tstart=self.tstart, tstep=self.tstep, tstop=self.tstop,
            stim_is_single=self.stim_is_single, stim_dims=self.stim_dims,
            stim_names=self.stim_names, baseline=self.baseline, scaling=self.scaling,
            stim_normalization=self.stim_normalization,
            basis_std=self.basis_std, sensor_dim=self.sensor_dim,
            is_whitened=True,
        )

    def timeslice(self, idx: Sequence[int] | IndexArray) -> RegressionData:
        """Return a new dataset restricted to selected time indices.

        If this dataset ``.is_whitened``, the returned dataset is also
        marked as whitened and quadratic forms are recomputed lazily.

        Parameters
        ----------
        idx
            Integer indices selecting the time samples to retain.
        """
        norm_factor = sqrt(len(idx))
        mul = self.norm_factor / norm_factor
        meg = [m[:, idx] * mul for m in self.meg]
        covariates = [c[idx, :] * mul for c in self.covariates]
        return RegressionData(
            meg, covariates, norm_factor,
            basis=self.basis,
            tstart=self.tstart, tstep=self.tstep, tstop=self.tstop,
            stim_is_single=self.stim_is_single, stim_dims=self.stim_dims,
            stim_names=self.stim_names, baseline=self.baseline, scaling=self.scaling,
            stim_normalization=self.stim_normalization,
            basis_std=self.basis_std, sensor_dim=self.sensor_dim,
            is_whitened=self.is_whitened,
        )


class NCRF:
    """Result container and object-based API for NCRF.

    Parameters
    ----------
    lead_field
        Forward solution a.k.a. lead-field matrix, with ``sensor`` and ``source``
        dimensions and an optional ``space`` dimension for free orientation.
    noise_covariance
        Noise covariance matrix in sensor space, typically estimated from empty-room
        recordings.
    n_iter
        Number of out iterations of the algorithm, by default set to 10.
    n_iterc
        Number of Champagne iterations within each outer iteration, by default set to 30.
    n_iterf
        Number of FASTA iterations within each outer iteration, by default set to 100.

    Attributes
    ----------
    h
        The neuro-current response function. It is one NDVar when fitting a single
        predictor and a sequence of NDVars when fitting multiple predictors.
    h_scaled
        ``h`` with the original stimulus scaling restored.
    explained_var
        Fraction of total variance explained by the fitted NCRFs.
    voxelwise_explained_variance
        Source-wise contributions to explained variance.
    Gamma
        Individual source covariance matrices.
    sigma_b
        Data covariance estimates under the model.
    theta
        NCRF coefficients over the Gabor basis.
    mu
        Regularization parameter used for the fitted model.
    residual
        The fit error, i.e. the result of the ``eval_obj`` error function on the
        final fit.
    tstart
        TRF start time in seconds, one value per predictor.
    tstep
        Sample spacing in seconds.
    tstop
        TRF stop time in seconds, one value per predictor.
    basis_std
        Standard deviation of the Gaussian basis functions in seconds.
    stim_baseline
        Mean that was subtracted from ``stim``.
    stim_scaling
        Scale by which ``stim`` was divided.

    Notes
    -----
    Usage:

    1. Use :meth:`RegressionData.from_data` to construct a prepared dataset
       from MEG and stimulus segments.
    2. Initialize :class:`NCRF` with the lead field and noise covariance.
    3. Call :meth:`NCRF.fit` with the :class:`RegressionData` instance to
       estimate the cortical TRFs.
    4. Access the cortical TRFs in the ``NCRF.h`` attribute.
    """
    _name = 'cTRFs estimator'
    _cv_results = None
    # Attributes to be assigned after fit:
    _data = None
    _stim_is_single = None
    _stim_dims = None
    _stim_names = None
    _stim_baseline = None
    _stim_scaling = None
    _stim_normalization = None
    _basis = None
    tstart = None
    tstep = None
    tstop = None
    explained_var = None
    _voxelwise_explained_variance = None
    residual = None
    mu = None
    theta = None
    basis_std = None
    tracker = None

    def __init__(
            self,
            lead_field: NDVar,
            noise_covariance: FloatArray,
            n_iter: int = 30,
            n_iterc: int = 10,
            n_iterf: int = 100,
    ) -> None:
        if lead_field.has_dim('space'):
            g = lead_field.get_data(dims=('sensor', 'source', 'space')).astype(np.float64)
            self.lead_field = g.reshape(g.shape[0], -1)
            self.space = lead_field.get_dim('space')
        else:
            g = lead_field.get_data(dims=('sensor', 'source')).astype(np.float64)
            self.lead_field = g
            self.space = None

        self.source = lead_field.get_dim('source')
        self.sensor = lead_field.get_dim('sensor')
        self.noise_covariance = noise_covariance.astype(np.float64)
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf
        self._prewhiten()

    def __repr__(self) -> str:
        if self.space:
            orientation = 'free'
        else:
            orientation = 'fixed'
        return f"<[{orientation} orientation] {self._name} on {self.source!r}>"

    def __copy__(self) -> NCRF:
        """Create a shallow copy with configuration but without fit results."""
        obj = type(self).__new__(self.__class__)
        copy_keys = ['lead_field', '_whitened_lead_field', 'lead_field_scaling', 'source', 'space', 'sensor',
                     '_whitening_filter', 'noise_covariance', '_whitened_noise_covariance',
                     'n_iter', 'n_iterc', 'n_iterf', 'eta', 'init_sigma_b', 'basis_std']
        for key in copy_keys:
            obj.__dict__.update({key: self.__dict__.get(key, None)})
        return obj

    # Primary attributes are sufficient to reconstruct the full object.  Derived
    # attributes (_whitening_filter, _whitened_lead_field, lead_field_scaling,
    # _whitened_noise_covariance) are recomputed by _prewhiten() and not stored.
    _PRIMARY_ATTRS = (
        '_basis', '_cv_results', 'mu', '_name', '_stim_is_single', '_stim_dims', '_stim_names',
        'noise_covariance', 'n_iter', 'n_iterc', 'n_iterf', 'lead_field', '_data',
        'explained_var', '_voxelwise_explained_variance', '_stim_baseline', '_stim_scaling',
        'residual', 'sensor', 'source', 'space', 'theta', 'tstart', 'tstep', 'tstop',
        'basis_std', '_stim_normalization', 'tracker'
    )

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the fitted model state needed for pickling."""
        return {k: getattr(self, k) for k in self._PRIMARY_ATTRS}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickled state while keeping compatibility with older versions."""
        for k in self._PRIMARY_ATTRS:
            setattr(self, k, state.get(k, None))
        if '_whitening_filter' in state:
            # Old pickle: noise_covariance was already whitened and lead_field may also
            # have been whitened+scaled.  Restore derived attrs directly.
            self._whitening_filter = state['_whitening_filter']
            self.lead_field_scaling = state.get('lead_field_scaling')
            self._whitened_noise_covariance = self.noise_covariance
            wlf = state.get('_whitened_lead_field')
            self._whitened_lead_field = wlf if wlf is not None else self.lead_field
        else:
            self._prewhiten()
        # make compatible with one tstop case
        if self._stim_dims is not None:
            self.tstop = self.tstop if isinstance(self.tstop, collections.abc.Sequence) else [self.tstop]
            self._basis = self._basis if isinstance(self._basis, collections.abc.Sequence) else [self._basis]
            if len(self._stim_dims) > 1:
                if len(self.tstop) != len(self._stim_dims):
                    self.tstop = self.tstop * len(self._stim_dims)
                if len(self._basis) != len(self._stim_dims):
                    self._basis = self._basis * len(self._stim_dims)
        # make compatible with the previous version
        if self._cv_results is None:
            info = state.get('_cv_info')
            if info is not None:
                _cv_results = []
                for items in info:
                    if isinstance(items, np.ndarray):
                        for columns in items.T:
                            _cv_results.append(CVResult(*columns[[0, 1, 4, 2, 3]]))
                setattr(self, '_cv_results', _cv_results)
        if self.basis_std is None:
            # Old bug: basis was always computed using std = 85 ms
            self.basis_std = 0.0085

    def _prewhiten(self) -> None:
        """Compute whitened derived quantities from ``lead_field`` and ``noise_covariance``.

        Writes to ``_whitening_filter``, ``_whitened_lead_field``,
        ``lead_field_scaling``, and ``_whitened_noise_covariance``.
        Neither ``lead_field`` nor ``noise_covariance`` is modified.
        """
        wf = _inv_sqrtm(self.noise_covariance)
        if (np.var(wf, axis=1) == 0).any():
            raise ValueError("Noise covariance data contains flat channels")
        self._whitening_filter = wf
        self._whitened_lead_field = np.dot(wf, self.lead_field)
        self._whitened_noise_covariance = wf.dot(self.noise_covariance).dot(wf.T)
        self.lead_field_scaling = linalg.norm(self._whitened_lead_field, 2)
        self._whitened_lead_field /= self.lead_field_scaling

    def _init_from_mne(self, data: RegressionData) -> None:
        """Seed source variances from a minimum-norm style initialization."""
        eta = []
        sigma_b = []
        dc = len(self.space) if self.space else 1
        for y, _ in data:
            t = y.shape[1]
            Gamma, data_cov = mne_initialization(y * (t ** 0.5), self._whitened_lead_field)
            Gamma = np.reshape(Gamma, (-1, dc))
            eta.append([np.diag(g) for g in Gamma])
            sigma_b.append(self._whitened_noise_covariance + data_cov)
        self.eta = eta
        self.init_sigma_b = sigma_b

    def _init_iter(self, data: RegressionData) -> None:
        """Initialize solver state for a new value of the regularization parameter."""
        if self.space:
            dc = len(self.space)
        else:
            dc = 1

        self.Gamma = []
        self.Sigma_b = []
        # for _ in range(len(data)):
        #     self.Gamma.append([self.eta * np.eye(dc, dtype=np.float64) for _ in range(len(self.source))])
        #     self.Sigma_b.append(self.init_sigma_b.copy())
        for g, s in zip(self.eta, self.init_sigma_b):
            self.Gamma.append(copy.deepcopy(g))
            self.Sigma_b.append(s.copy())

        # initializing \Theta
        l = sum([basis.shape[1] * (len(dim) if dim else 1) for basis, dim in zip(data.basis, data.stim_dims)])
        self.theta = np.zeros((len(self.source) * dc, l), dtype=np.float64)

    def _set_mu(self, mu: float, data: RegressionData) -> None:
        """Reset the solver state for the requested regularization value."""
        self.mu = mu
        self._init_iter(data)
        if mu == 0.0:
            self._solve(data, self.theta, n_iterc=30)

    def _solve(
            self,
            data: RegressionData,
            theta: FloatArray,
            idx: slice | IndexArray = slice(None, None),
            n_iterc: int | None = None,
    ) -> None:
        """Champagne steps implementation

        Parameters
        ----------
        data
            Whitened regression data to fit.
        theta
            Coefficients of the TRFs over the Gabor basis.

        Notes
        -----
        Implementation details can be found at:
        D. P. Wipf, J. P. Owen, H. T. Attias, K. Sekihara, and S. S. Nagarajan,
        “Robust Bayesian estimation of the location, orientation, and time course
        of multiple correlated neural sources using MEG,” NeuroImage, vol. 49,
        no. 1, pp. 641–655, 2010
        """
        logger = logging.getLogger('Champagne')
        # Choose dc
        if self.space:
            dc = len(self.space)
        else:
            dc = 1

        if n_iterc is None:
            n_iterc = self.n_iterc

        logger.debug('Champagne Iterations start:')
        logger.debug('trial \t time taken')
        for key, (meg, covariates) in enumerate(data):
            start = time.time()
            meg = meg[idx]
            covariates = covariates[idx]
            y = meg - np.dot(np.dot(self._whitened_lead_field, theta), covariates.T)
            Cb = np.dot(y, y.T)  # empirical data covariance

            try:
                raise np.linalg.LinAlgError
                yhat = linalg.cholesky(Cb, lower=True)
            except np.linalg.LinAlgError:
                hi = y.shape[0] - 1
                lo = max(y.shape[0] - y.shape[1], 0)
                e, v = linalg.eigh(Cb, subset_by_index=(lo, hi))
                tol = e[-1] * _R_tol
                indices = e > tol
                yhat = v[:, indices] * np.sqrt(e[indices])[None, :]

            # gamma = self.Gamma[key].copy()
            # sigma_b = self.Sigma_b[key].copy()
            gamma = copy.deepcopy(self.eta[key])
            sigma_b = self.init_sigma_b[key].copy()

            # champagne iterations
            for it in range(n_iterc):
                # pre-compute some useful matrices
                try:
                    Lc = linalg.cholesky(sigma_b, lower=True)
                    lhat = linalg.solve(Lc, self._whitened_lead_field)
                    ytilde = linalg.solve(Lc, yhat)
                except np.linalg.LinAlgError:
                    Lc = _inv_sqrtm(sigma_b)
                    lhat = np.dot(Lc, self._whitened_lead_field)
                    ytilde = np.dot(Lc, yhat)

                # compute sigma_b for the next iteration
                sigma_b[:] = self._whitened_noise_covariance[:]
                # tempx = lhat.T @ ytilde

                for i in range(len(self.source)):
                    if dc > 1:
                        # update Xi
                        x = np.dot(gamma[i], np.dot(lhat[:, i * dc:(i + 1) * dc].T, ytilde))
                        # x = np.dot(gamma[i], tempx[i * dc:(i + 1) * dc, :])
                        # update Zi
                        z = np.dot(lhat[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])
                    else:
                        # update Xi
                        x = gamma[i] * lhat[:, i].T.dot(ytilde)
                        # x = gamma[i] * tempx[i]
                        # update Zi
                        z = (lhat[:, i] ** 2).sum()

                    # update Ti
                    if dc == 1:
                        # gamma[i] = sqrt(np.einsum('i,i->',x, x)) / np.real(sqrt(z))
                        gamma[i] = sqrt((x ** 2).sum()) / np.real(sqrt(z))
                    elif dc == 3:
                        _compute_gamma_ip(z, x, gamma[i])
                    else:
                        gamma[i] = _compute_gamma_i(z, x)

                    # update sigma_b for next iteration
                    sigma_b += np.dot(self._whitened_lead_field[:, i * dc:(i + 1) * dc],
                                      np.dot(gamma[i], self._whitened_lead_field[:, i * dc:(i + 1) * dc].T))

            self.Gamma[key] = gamma
            self.Sigma_b[key] = sigma_b
            end = time.time()
            logger.debug(f'{key} \t {end - start}')

    def fit(
            self,
            data: RegressionData,
            mu: MuArg = 'auto',
            do_crossvalidation: bool = False,
            tol: float = 1e-5,
            verbose: bool = False,
            use_ES: bool = False,
            mus: MusArg = None,
            n_splits: int = None,
            n_workers: int = None,
            compute_explained_variance: bool = False,
            accept_whitening: bool = False,
            track_progress: bool = False,
    ) -> None:
        """Fit the NCRF model to prepared regression data.

        Estimate both TRFs and source variance from the observed MEG data by solving
        the Bayesian optimization problem formulated in :cite:`das2020neuro`.

        Parameters
        ----------
        data
            M/EEG data and the corresponding stimulus variables.
        mu
            Regularization parameter; promote sparsity and guard against over-fitting
        do_crossvalidation
            if True, from a wide range of regularizing parameters, the one resulting in
            the least generalization error in a k-fold cross-validation procedure is chosen.
            Unless specified the range and k is chosed from cofig.py. The user can also pass
            several keyword arguments to overwrite them.
        tol
            tolerence parameter. Decides when to stop outer iterations.
        verbose
            If set True prints intermediate values of the cost functions (default ``False``).
        use_ES
            use estimation stability criterion :cite:`limEstimationStabilityCrossValidation2016`
            to choose the best ``mu`` (default ``False``).
        mus
            range of mu to be considered for cross-validation
        n_splits
            k value used in k-fold cross-validation
        n_workers
            Number of workers to use for cross-validation.
            ``None`` to use ``cpu_count/2`` (default).
            ``0`` to run without :mod:`multiprocessing`.
        compute_explained_variance
            Compute voxel-wise explained variance.
        accept_whitening
            Accept pre-whitened data. This is intended for internal workflows
            that slice an already-whitened dataset, such as cross-validation.
        """
        logger = logging.getLogger(__name__)
        if data.is_whitened:
            if not accept_whitening:
                raise ValueError("data is already whitened; pass accept_whitening=True to accept it")
        else:
            data = data.whiten(self._whitening_filter)

        tracker = OptimizationTracker() if track_progress else None

        logger.info('Initiating from mne sol, please wait...')
        self._init_from_mne(data)
        logger.info('Thanks for waiting...')

        # take care of cross-validation
        if do_crossvalidation:
            if mus == 'auto':
                mus = self._auto_mu(data)
            logger.info('Crossvalidation initiated!')
            cv_results = crossvalidate(self, data, mus, tol, n_splits, n_workers)
            best_cv = min(cv_results, key=attrgetter('cross_fit'))
            if best_cv.mu == min(mus):
                logger.info(f'CVmu is {best_cv.mu}: extending range of mu towards left')
                new_mus = np.logspace(np.log10(best_cv.mu) - 1, np.log10(best_cv.mu), 4)[:-1]
            elif best_cv.mu == max(mus):
                logger.info(f'CVmu is {best_cv.mu}: extending range of mu towards right')
                new_mus = np.logspace(np.log10(best_cv.mu), np.log10(best_cv.mu) + 1, 4)[1:]
            else:
                new_mus = None

            if new_mus is not None:
                cv_results.extend(crossvalidate(self, data, new_mus, tol, n_splits, n_workers))

            self._cv_results = cv_results
            best_cv = min(cv_results, key=attrgetter('cross_fit'))
            mu = best_cv.mu
            if use_ES:
                cv_results_ = sorted(self._cv_results, key=attrgetter('mu'))
                if mu == cv_results[-1].mu:
                    logger.info(f'\nCVmu is {best_cv.mu}: could not find mu based on estimation stability criterion\nContinuing with cross-validation only.')
                else:
                    best_es = None
                    for i, res in enumerate(cv_results_):
                        if res.mu < mu:
                            continue
                        else:
                            try:
                                if res.estimation_stability < cv_results_[i + 1].estimation_stability:
                                    best_es = res
                                    break
                            except IndexError:
                                best_es = None
                    if best_es is None:
                        logger.warning('\nNo ES minima found: could not find mu based on estimation stability criterion.\nContinuing with cross-validation only.')
                    else:
                        mu = best_es.mu

        elif mu is None:  # use the passed mu
            raise TypeError(f'{mu=}: mu needs mu to be a number or "auto"')

        self._set_mu(mu, data)

        if self.space:
            def g_funct(x): return g_group(x, self.mu)
            def prox_g(x, t): return proxg_group_opt(x, self.mu * t)
        else:
            def g_funct(x): return g(x, self.mu)
            def prox_g(x, t): return shrink(x, self.mu * t)

        theta = self.theta

        myname = current_process().name

        self.err = []
        self.objective_vals = []
        if verbose:
            iter_o = tqdm(range(self.n_iter))
        else:
            iter_o = range(self.n_iter)

        logger.debug('process:iteration \t objective value \t %% change')
        # run iterations
        for i in iter_o:
            funct, grad_funct = self._construct_f(data)
            logger.debug(f"Before FASTA:{funct(self.theta)}")
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.n_iterf)
            Theta.learn(theta)

            self.err.append(self._residual(theta, Theta.coefs_))
            theta = Theta.coefs_
            self.theta = theta
            logger.debug(f"After FASTA: {funct(self.theta)}")

            if self.err[-1] < tol:
                break

            self._solve(data, theta)

            self.objective_vals.append(self.eval_obj(data))

            if tracker is not None:
                tracker.record(i, self.objective_vals[-1], self.err[-1], self.theta)

            logger.debug(f'{myname}:{i} \t {self.objective_vals[-1]} \t {self.err[-1] * 100}')

        self.residual = self.eval_obj(data)
        self._copy_from_data(data)
        self.explained_var = self.compute_explained_variance(data)
        if compute_explained_variance:
            self._voxelwise_explained_variance = self._compute_voxelwise_explained_variance(data)
        self._data = data  # save the data for further use
        self.tracker = tracker

    def _copy_from_data(self, data: RegressionData) -> None:
        """Copy stimulus metadata needed to rebuild Eelbrain output objects."""
        self._stim_is_single = data.stim_is_single
        self._stim_dims = data.stim_dims
        self._stim_names = data.stim_names
        self._stim_baseline = data.baseline
        self._stim_scaling = data.scaling
        self._stim_normalization = data.stim_normalization
        self._basis = data.basis
        self.tstart = data.tstart
        self.tstep = data.tstep
        self.tstop = data.tstop
        self.basis_std = data.basis_std

    def _construct_f(self, data: RegressionData) -> tuple[ObjectiveFunction, GradientFunction]:
        """Build the smooth objective and gradient passed to FASTA.

        Parameters
        ----------
        data
            Prepared regression data.
        """
        leadfields = []
        bEs = []
        bbts = []
        for i in range(len(data)):
            try:
                raise np.linalg.LinAlgError
                L = linalg.cholesky(self.Sigma_b[i], lower=True)
                leadfields.append(linalg.solve(L, self._whitened_lead_field))
                bEs.append(linalg.solve(L, data.bE[i]))
                bbts.append(np.trace(linalg.solve(L, linalg.solve(L, data.bbt[i]).T)))
            except np.linalg.LinAlgError:
                Linv = _inv_sqrtm(self.Sigma_b[i])
                leadfields.append(np.dot(Linv, self._whitened_lead_field))
                bEs.append(np.dot(Linv, data.bE[i]))
                bbts.append(np.trace(np.dot(Linv, np.dot(Linv, data.bbt[i]).T)))

        def f(L, x, bbt, bE, EtE):
            Lx = np.dot(L, x)
            y = bbt - 2 * np.sum(bE * Lx) + np.sum(Lx * np.dot(Lx, EtE))
            return 0.5 * y

        def gradf(L, x, bE, EtE):
            y = bE - np.dot(np.dot(L, x), EtE)
            return -np.dot(L.T, y)

        def funct(x):
            fval = 0.0
            for i in range(len(data)):
                fval = fval + f(leadfields[i], x, bbts[i], bEs[i], data.EtE[i])
            return fval

        def grad_funct(x):
            grad = gradf(leadfields[0], x, bEs[0], data.EtE[0]).astype(np.float64)
            for i in range(1, len(data)):
                grad += gradf(leadfields[i], x, bEs[i], data.EtE[i])
            return grad

        return funct, grad_funct

    def eval_obj(
            self,
            data: RegressionData,
            return_wl2: bool = False,
    ) -> float | tuple[float, float]:
        """Evaluate the current objective value on a dataset.

        Parameters
        ----------
        data
            Dataset on which to evaluate the objective.

        Returns
        -------
        float | tuple[float, float]
            Objective value, or a pair containing the objective value and the
            weighted L2 term when ``return_wl2`` is true.
        """
        ll2 = 0
        logdet = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.dot(np.dot(self._whitened_lead_field, self.theta), covariate.T)
            Cb = np.dot(y, y.T)  # empirical data covariance
            try:
                yhat = linalg.cholesky(Cb, lower=True)
            except np.linalg.LinAlgError:
                hi = y.shape[0] - 1
                lo = max(y.shape[0] - y.shape[1], 0)
                e, v = linalg.eigh(Cb, subset_by_index=(lo, hi))
                tol = e[-1] * _R_tol
                indices = e > tol
                yhat = v[:, indices] * np.sqrt(e[indices])

            # L = linalg.cholesky(self.Sigma_b[key], lower=True)
            # y = linalg.solve(L, yhat)
            sigma_b = self.Sigma_b[key]
            try:
                Lc = linalg.cholesky(sigma_b, lower=True)
                y = linalg.solve(Lc, yhat)
                logdet_ = np.log(np.diag(Lc)).sum()
            except np.linalg.LinAlgError:
                Lc, e = _inv_sqrtm(sigma_b, return_eig=True)
                y = np.dot(Lc, yhat)
                logdet_ = -np.log(e).sum()

            ll2 += 0.5 * (y ** 2).sum()
            logdet += logdet_
        if return_wl2:
            return (ll2 + logdet) / len(data), ll2 / len(data)
        return (ll2 + logdet) / len(data)

    def eval_l2(self, data: RegressionData) -> float:
        """Evaluate the unweighted L2 prediction error used in CV."""
        l2 = 0
        for key, (meg, covariate) in enumerate(data):
            y = meg - np.dot(np.dot(self._whitened_lead_field, self.theta), covariate.T)
            l2 += 0.5 * (y ** 2).sum()  # + np.log(np.diag(L)).sum()

        return l2 / len(data)

    def compute_explained_variance(self, data: RegressionData) -> float:
        """Compute the global explained-variance score for a fitted model."""
        logger = logging.getLogger('NCRF: Explained Variance')
        temp = 0
        for key, (meg, covariate) in enumerate(data):
            # W = _inv_sqrtm(self.Sigma_b[key])
            # W_meg = W @ meg
            # W_leadfield = W @ self._whitened_lead_field
            W_meg = meg
            W_leadfield = self._whitened_lead_field
            y = W_meg - np.dot(np.dot(W_leadfield, self.theta), covariate.T)
            # temp += (y * y).sum() / (W_meg * W_meg).sum()  # + np.log(np.diag(L)).sum()
            temp += np.nansum(np.var(y, axis=1) / np.var(W_meg, axis=1)) / y.shape[0]

        logger.debug(f'{self.mu}: {1 - temp / len(data)}')
        return 1 - temp / len(data)

    def _compute_voxelwise_explained_variance(self, data: RegressionData) -> FloatArray:
        """Compute each source's contribution to explained variance."""
        temp = np.zeros(len(self.source))
        theta = self.theta.copy()
        for key, (meg, covariate) in enumerate(data):
            # W = _inv_sqrtm(self.Sigma_b[key])
            # W_meg = W @ meg
            # W_leadfield = W @ self._whitened_lead_field
            W_meg = meg
            W_leadfield = self._whitened_lead_field
            total_var = np.var(W_meg, axis=1)
            y = W_meg - np.dot(np.dot(W_leadfield, theta), covariate.T)
            explained_variance = np.var(y, axis=1)
            for i, _ in enumerate(self.source):
                theta[:] = self.theta[:]
                if self.space is None:
                    theta[i] = 0
                else:
                    theta[i * len(self.space):(i + 1) * len(self.space)] = 0
                y = W_meg - np.dot(np.dot(W_leadfield, theta), covariate.T)
                temp[i] += np.nansum((np.var(y, axis=1) - explained_variance) / total_var) / W_meg.shape[0]  # + np.log(
                # np.diag(L)).sum()

        return temp / len(data)

    @cached_property
    def voxelwise_explained_variance(self) -> NDVar | None:
        """Voxelwise explained variance expressed on the source dimension."""
        if self._voxelwise_explained_variance is None:
            return None
        else:
            return NDVar(self._voxelwise_explained_variance, self.source)

    @cached_property
    def h_scaled(self) -> NDVar | list[NDVar]:
        """Return ``h`` with the original stimulus scaling restored."""
        if self._stim_scaling is None:
            return self.h
        elif self._stim_is_single:
            return self.h * self._stim_scaling[0]
        else:
            return [h * s for h, s in zip(self.h, self._stim_scaling)]

    @cached_property
    def h(self) -> NDVar | list[NDVar]:
        """Return the spatio-temporal response function as Eelbrain NDVars."""
        n_vars = sum(len(dim) if dim else 1 for dim in self._stim_dims)
        if self.space:
            _shared_dims = (self.source, self.space)
        else:
            _shared_dims = (self.source, )

        if n_vars > 1:
            _trf = []
            start = 0
            stop = 0
            for basis, dim in zip(self._basis, self._stim_dims):
                stim_len = len(dim) if dim else 1
                stop += basis.shape[1] * stim_len
                theta = self.theta[:, start:stop].copy()
                shape = (self.theta.shape[0], stim_len, -1)
                theta = theta.reshape(shape)
                _trf.append(np.squeeze(theta.swapaxes(1, 0)))
                start += basis.shape[1] * stim_len
        else:
            _trf = [self.theta]

        trf = [np.dot(x, basis.T) / self.lead_field_scaling for x, basis in zip(_trf, self._basis)]

        h = []
        for x, dim, name, tstart in zip(trf, self._stim_dims, self._stim_names, self.tstart):
            if dim:
                time = UTS(tstart, self.tstep, x.shape[-1])
                shared_dims = (*_shared_dims, time)
                x = x.reshape((-1, *(map(len, shared_dims))))
                dims = (dim, *shared_dims)
            else:
                time = UTS(tstart, self.tstep, x.shape[-1])
                dims = (*_shared_dims, time)
                x = x.reshape(*(map(len, dims)))
            h.append(NDVar(x, dims, name=name))

        if self._stim_is_single:
            return h[0]
        else:
            return h

    @staticmethod
    def _residual(theta0: FloatArray, theta1: FloatArray) -> float:
        diff = theta1 - theta0
        num = diff ** 2
        den = theta0 ** 2
        if den.sum() <= 0:
            return np.inf
        else:
            return sqrt(num.sum() / den.sum())

    @staticmethod
    def compute_ES_metric(models: Sequence[NCRF], data: RegressionData) -> float:
        """Compute the estimation-stability metric across cross-validation folds.

        Details can be found at:
        Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
        Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

        Parameters
        ----------
        models
            Fitted models from different cross-validation folds.
        data
            Dataset used to compare their predictions.

        Returns
        -------
        float
            Estimation-stability score.
        """
        Y = []
        for model in models:
            y = np.empty(0)
            for trial in range(len(data)):
                y = np.append(y, np.dot(np.dot(model._whitened_lead_field, model.theta), data.covariates[trial].T))
            Y.append(y)
        Y = np.array(Y)
        Y_bar = Y.mean(axis=0)
        VarY = (((Y - Y_bar) ** 2).sum(axis=1)).mean()
        if (Y_bar ** 2).sum() <= 0:
            return np.inf
        else:
            return VarY / (Y_bar ** 2).sum()

    def cvfunc(self, data: RegressionData, n_splits: int, tol: float, mu: float) -> CVResult:
        cvfun = self._get_cvfunc(data, n_splits, tol)
        return cvfun(mu)

    def _get_cvfunc(
            self,
            data: RegressionData,
            n_splits: int,
            tol: float,
    ) -> Callable[[float], CVResult]:
        """Create the callable executed by cross-validation workers.

        In the cross-validation phase the workers will call this function for
        different regularizer parameters.

        Parameters
        ----------
        data
            Dataset object compatible with model fitting and exposing
            :meth:`ncrf.RegressionData.timeslice` for train/test partitioning.
        n_splits
            number of folds for cross-validation, If None, it will use values
            specified in config.py.
        tol
            tolerence parameter. Decides when to stop outer iterations.

        Returns
        -------
        callable
            Callable that evaluates one regularization value and returns the
            cross-validation metrics.
        """
        models_ = [copy.copy(self) for _ in range(n_splits)]
        # from sklearn.model_selection import KFold
        from ._crossvalidation import TimeSeriesSplit

        def cvfunc(mu: float) -> CVResult:
            # kf = KFold(n_splits=n_splits)
            d = max(basis.shape[1] for basis in data.basis)
            kf = TimeSeriesSplit(r=0.05, p=n_splits, d=d)
            ll = []
            ll1 = []
            ll2 = []
            for model_, (train, test) in zip(models_, kf.split(data.meg[0][0])):
                traindata = data.timeslice(train)
                testdata = data.timeslice(test)
                model_.fit(traindata, mu, tol=tol, verbose=False, accept_whitening=True)
                obj, wl2 = model_.eval_obj(testdata, True)
                ll.append(wl2)
                ll1.append(obj)
                ll2.append(model_.eval_l2(testdata))

            time.sleep(0.001)
            return CVResult(
                mu,
                sum(ll) / len(ll),  # weighted_l2_error
                self.compute_ES_metric(models_, data),  # estimation_stability
                sum(ll1) / len(ll1),  # cross_fit
                sum(ll2) / len(ll2),  # l2_error
            )

        return cvfunc

    def _auto_mu(self, data: RegressionData, p: float = 99.0) -> FloatArray:
        """Infer a candidate regularization grid from the gradient magnitudes."""
        self._set_mu(0.0, data)
        _, grad_funct = self._construct_f(data)
        if self.space:
            x = grad_funct(self.theta)
            l = x.shape[1]
            x.shape = (-1, 3, l)
            norm = np.linalg.norm(x, axis=1)
        else:
            x = grad_funct(self.theta)
            norm = np.abs(x)

        hi = log10(np.percentile(norm, p))
        lo = hi - 2
        return np.logspace(lo, hi, 7)

    def cv_info(self) -> fmtxt.Table:
        """Summarize stored cross-validation scores in a table."""
        if self._cv_results is None:
            raise ValueError("CV: no cross-validation was performed. Use mu='auto' to perform cross-validation.")
        cv_results = sorted(self._cv_results, key=attrgetter('mu'))
        criteria = ('cross-fit', 'l2/mu')
        best_mu = {criterion: self.cv_mu(criterion) for criterion in criteria}

        table = fmtxt.Table('lllll')
        table.cells('mu', 'cross-fit', 'l2-error', 'weighted l2-error', 'ES metric')
        table.midrule()
        fmt = '%.5f'
        for result in cv_results:
            table.cell(fmtxt.stat(result.mu, fmt=fmt))
            star = 1 if result.mu is best_mu['cross-fit'] else 0
            table.cell(fmtxt.stat(result.cross_fit, fmt, star, 1))
            star = 1 if result.mu is best_mu['l2/mu'] else 0
            table.cell(fmtxt.stat(result.l2_error, fmt, star, 1))
            table.cell(fmtxt.stat(result.weighted_l2_error, fmt=fmt))
            table.cell(fmtxt.stat(result.estimation_stability, fmt=fmt))
        # warnings
        mus = [res.mu for res in self._cv_results]
        warnings = []
        if self.mu == min(mus):
            warnings.append("Best mu is smallest mu")
        if warnings:
            table.caption(f"Warnings: {'; '.join(warnings)}")
        return table

    def cv_mu(self, criterion: str = 'cross-fit') -> float:
        """Retrieve best mu based on cross-validation

        Parameters
        ----------
        criterion
            Criterion for best fit. Possible values:

            - ``'cross-fit'``: The smallest cross-fit value (default)
            - ``'l2'``: The smallest l2 error
            - ``'l2/mu'``: The local minimum in the l2 error with smallest mu
        """
        if criterion == 'cross-fit':
            best_cv = min(self._cv_results, key=attrgetter('cross_fit'))
        elif criterion == 'l2':
            best_cv = min(self._cv_results, key=attrgetter('l2_error'))
        elif criterion == 'l2/mu':
            cv_results = sorted(self._cv_results, key=attrgetter('mu'))
            peaks, _ = find_peaks([-result.l2_error for result in cv_results])  # find local minima
            if len(peaks) > 0:
                # higher mu -> smaller trf
                best_cv = max([cv_results[peak] for peak in peaks], key=attrgetter('mu'))
            else:
                best_cv = min(cv_results, key=attrgetter('l2_error'))
        else:
            raise ValueError(f'criterion={criterion}')
        return best_cv.mu


# Functions used for initialize \Gamma
def find_mu(
        s: FloatArray,
        y: FloatArray,
        eta: float = 1,
        tol: float = 1e-8,
        max_iter: int = 1000,
) -> float:
    """Solve for the empirical-Bayes noise parameter used in initialization."""
    logger = logging.getLogger(__name__)
    e = s ** 2
    z = y ** 2
    TM = z.size
    eta = eta * TM
    z2 = z.sum(axis=1)
    mu = 0
    diff = []

    logger.info('please wait: calculating mu...')
    for _ in range(max_iter):
        temp = 1 + mu * e
        fmu = z2 / (temp ** 2)
        f = fmu.sum() - eta
        dfmu = (-2) * fmu * e / temp
        diff.append(f / dfmu.sum())
        if (mu == 0 and f < 0) or abs(diff[-1] / diff[0]) < tol:
            logger.info(f"thanks for waiting, (mu: {mu}) calculation complete after:"
                        f"iteration # {len(diff)} with relative error {diff[-1] / diff[0]}")
            return mu
        mu -= diff[-1]

    logger.info(f"maximum iteration {max_iter} reached, consider more iterations for convergence!")
    return mu


def wls(
        y: FloatArray,
        l: FloatArray,
        w: FloatArray,
        return_ecov: bool = False,
) -> tuple[FloatArray, float] | tuple[FloatArray, float, FloatArray]:
    """Solve the weighted least-squares problem used for NCRF initialization."""
    w = np.squeeze(w)
    if w.ndim == 1:
        lw = l * w[None, :]
    else:
        lw = l @ w
    u, s, vh = linalg.svd(lw, full_matrices=False)
    yw = u.T @ y
    mu = find_mu(s, yw, eta=1)
    if mu:
        gamma = s / (s ** 2 + 1 / mu)
    else:
        gamma = 1 / s

    if w.ndim == 1:
        im = w[:, None] * vh.T
    else:
        im = w @ vh.T

    im = im * gamma[None, :]

    if return_ecov is True:
        ecov = np.eye(w.shape[0]) - vh.T @ ((gamma * s)[:, None] * vh)
        ecov *= mu
        if w.ndim == 1:
            ecov *= w[:, None]
            ecov *= w[None, :]
        else:
            ecov = ecov @ w.T
            ecov = w @ ecov
        return im @ yw, mu, ecov

    return im @ yw, mu


def mne_initialization(
        y: FloatArray,
        l: FloatArray,
        use_depth_prior: bool = True,
        exp: float = 0.8,
) -> tuple[FloatArray, FloatArray]:
    """Build the initial Gamma and sensor covariance from an MNE-style estimate."""
    N, M = l.shape
    T = y.shape[1]

    if use_depth_prior:
        dw = 1.0 / (l ** 2).sum(axis=0)
        limit = dw.min() * 10.0
        depth_weighting = (np.minimum(dw / limit, 1)) ** exp
    else:
        depth_weighting = np.ones(M)

    w = np.ones(M)
    w *= depth_weighting
    inv, mu, ecov = wls(y, l, w, return_ecov=True)
    Gamma = np.diag((inv @ inv.T) / T + ecov)
    data_cov = l * Gamma[None, :] @ l.T
    return Gamma, data_cov
