"""Prepared regression dataset and covariate construction for NCRF fitting.

``RegressionData`` turns Eelbrain objects into normalized numeric arrays with a
stable internal layout that the solver consumes directly. The layout itself is
described by the ``TRFDesign`` it carries.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from math import sqrt
from typing import Any
from collections.abc import Iterator, Sequence

from eelbrain import NDVar, Sensor, UTS
import numpy as np
import numpy.typing as npt
from scipy import linalg

from ._trf_design import TRFDesign, stim_dimensions
from ._pickle import pickle_state
from ._repr import _count_repr
from ._typing import FloatArray, IndexArray, ScaleArg, TrialData


SCALES = ('l1', 'l2', 'spectral')


def get_scaling(
        stim: Sequence[Sequence[NDVar]],
        design: TRFDesign,
        scale: ScaleArg,
) -> tuple[FloatArray, FloatArray | None]:
    """Stimulus centering and scaling values, one per expanded covariate channel.

    Parameters
    ----------
    stim
        Stimulus lists, one per segment; each inner list contains one NDVar per
        predictor.
    design
        Design describing the predictors.
    scale
        Compute the ``'l1'`` (mean absolute deviation) or ``'l2'`` (standard
        deviation) scale of each predictor. Any other value yields ``None`` for the
        scaling, since it is then not derived from the stimulus. The ``'spectral'``
        scale is not a property of the predictor and is computed from the
        covariates instead (see :meth:`RegressionData._spectral_norms`).

    Returns
    -------
    baseline
        The mean of each predictor.
    scaling
        The requested scale of each predictor, measured around its mean, or
        ``None``.

    Raises
    ------
    ValueError
        If a predictor channel is constant over time, and hence has no variation
        to scale by.
    """
    by_predictor = list(zip(*stim))  # -> [[stim_1_trial_1, stim_1_trial_2, ...], ...]
    _assert_varying(by_predictor, design)

    n = sum(len(x.time) for x in by_predictor[0])
    means = [sum(x.sum('time') for x in trials) / n for trials in by_predictor]
    baseline = _channel_values(means, design.stim_lens)

    # Scale by the variation around the mean, whether or not the covariates end up
    # centered; the raw magnitude would let a predictor's offset dominate its scale
    centered = [[x - mean for x in trials] for mean, trials in zip(means, by_predictor)]
    if scale == 'l1':
        scales = [sum(x.abs().sum('time') for x in trials) / n for trials in centered]
    elif scale == 'l2':
        scales = [(sum((x ** 2).sum('time') for x in trials) / n) ** 0.5 for trials in centered]
    else:  # 'spectral' is computed after covariate construction
        return baseline, None
    return baseline, _channel_values(scales, design.stim_lens)


def _assert_varying(
        by_predictor: Sequence[Sequence[NDVar]],
        design: TRFDesign,
) -> None:
    """Check that every predictor channel varies over time.

    Parameters
    ----------
    by_predictor
        Segments of each predictor.
    design
        Design the predictors belong to, used to name the offending channels.

    Raises
    ------
    ValueError
        If a predictor channel has the same value at every time point.
    """
    constant = []
    for name, trials in zip(design.stim_names, by_predictor):
        # (n_channels, n_times) per segment
        arrays = [np.atleast_2d(t.get_data(t.get_dimnames(last='time'))) for t in trials]
        lo = np.min([x.min(1) for x in arrays], axis=0)
        hi = np.max([x.max(1) for x in arrays], axis=0)
        for i in np.flatnonzero(lo == hi):
            desc = name or '<unnamed>'
            if len(lo) > 1:
                desc = f'{desc}[{i}]'
            constant.append(desc)
    if constant:
        raise ValueError(f"{', '.join(constant)}: predictor is constant over time, so it has no variation to scale by; drop it, or prepare the data with scale=None")


def _channel_values(
        values: Sequence[NDVar | float],
        stim_lens: Sequence[int],
) -> FloatArray:
    """Flatten one value per predictor into one value per covariate channel."""
    return np.concatenate([x.x if isinstance(x, NDVar) else np.full(n, x) for x, n in zip(values, stim_lens)])


def _check_scaling(
        scaling: FloatArray,
        design: TRFDesign,
) -> None:
    """Check that scaling factors can be divided by without destroying the covariates.

    Dividing by a factor of 0 fills the covariate columns with NaN, and a negative
    or non-finite factor corrupts them just as silently, so a design carrying such
    factors must be rejected rather than applied.

    Parameters
    ----------
    scaling
        Scaling factor of each covariate channel.
    design
        Design the factors belong to, used to name the offending channels.

    Raises
    ------
    ValueError
        If any factor is not finite and strictly positive.
    """
    bad = ~(np.isfinite(scaling) & (scaling > 0))
    if not bad.any():
        return
    channels = [name if n == 1 else f'{name}[{i}]' for name, n in zip(design.stim_names, design.stim_lens) for i in range(n)]
    items = ', '.join(f'{channels[i]}={scaling[i]:g}' for i in np.flatnonzero(bad))
    raise ValueError(f"invalid {design.scale!r} scaling ({items}): scaling factors must be finite and > 0; a predictor that is constant over time has no variation to scale by")


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

    Y = []
    for w, start, M in zip(ws, starts, Ms):
        # X[i, j] = w[i - j], zero-padded where the history runs past the start
        X = linalg.toeplitz(w, np.zeros(M, dtype=w.dtype))
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


def _project_basis(
        raw_covs: Sequence[FloatArray],
        design: TRFDesign,
        norm_factor: float,
) -> FloatArray:
    """Project each channel's lag matrix onto its predictor's Gaussian basis.

    Parameters
    ----------
    raw_covs
        Lag matrix of each expanded covariate channel, in design order.
    design
        Design supplying the basis of each predictor.
    norm_factor
        Value the covariates are divided by, matching the MEG data.

    Returns
    -------
    ndarray
        Covariate matrix, shape ``(n_times, n_basis_cols)``.
    """
    covariates = []
    i = 0
    for n, basis in zip(design.stim_lens, design.basis):
        covariates.extend(np.dot(x, basis) / norm_factor for x in raw_covs[i:i + n])
        i += n
    return np.concatenate(covariates, axis=1).astype(np.float64)


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
    design
        The ``TRFDesign`` that ``covariates`` were built with. It also records the
        normalization that was applied to ``covariates`` (see :meth:`normalize`),
        which is what lets a model fitted on one dataset predict another.
    sensor_dim
        Sensor dimension shared by all MEG segments.
    whitener
        The whitening filter that was applied to ``meg``, or ``None`` for raw
        data. Recording the filter itself lets :meth:`whiten` distinguish a
        no-op (same filter) from an error (different filter).
    """

    meg: list[FloatArray]  # (sensor, time)
    covariates: list[FloatArray]  # (time, covariate)
    design: TRFDesign
    sensor_dim: Sensor
    whitener: FloatArray | None = None

    @property
    def norm_factor(self) -> float:
        """``sqrt(n_times)`` that MEG and covariates are divided by."""
        return sqrt(self.meg[0].shape[1])

    @property
    def is_whitened(self) -> bool:
        """Whether ``meg`` has been transformed by a whitening filter (see ``whitener``)."""
        return self.whitener is not None

    def __post_init__(self) -> None:
        # Direct construction bypasses from_data(), so check that the arrays fit
        # the metadata; normalize() and the solvers rely on covariates that are
        # projected onto the design's basis
        if len(self.meg) != len(self.covariates):
            raise ValueError(f"{_count_repr(len(self.meg), 'MEG segment')} but {_count_repr(len(self.covariates), 'covariate matrix', 'covariate matrices')}")
        if len({m.shape[1] for m in self.meg}) > 1:
            raise NotImplementedError("Segments with unequal trial length")
        n_sensors = len(self.sensor_dim)
        n_coefficients = self.design.n_coefficients
        for i, (m, cov) in enumerate(zip(self.meg, self.covariates)):
            if m.shape[0] != n_sensors:
                raise ValueError(f"segment {i}: MEG has {_count_repr(m.shape[0], 'channel')}, sensor_dim {_count_repr(n_sensors, 'sensor')}")
            elif cov.shape != (m.shape[1], n_coefficients):
                raise ValueError(f"segment {i}: covariates have shape {cov.shape}, expected {(m.shape[1], n_coefficients)} (n_times, n_coefficients); were they projected onto the basis?")

    @classmethod
    def from_data(
            cls,
            meg: list[NDVar],
            stim: list[Sequence[NDVar]],
            tstart: float | Sequence[float],
            tstop: float | Sequence[float],
            basis_stride: int = 1,
            scale: ScaleArg = 'spectral',
            stim_is_single: bool = False,
            basis_std: float = 0.0085,
            in_place: bool = False,
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
        basis_stride
            Spacing between neighboring Gabor basis atoms, in samples: with the
            default of ``1`` the atoms are one sample apart, and larger values
            make the basis sparser. ``basis_stride > 2`` should be used with caution.
        scale
            Normalization applied to the covariates. Each predictor's mean is
            subtracted, and each covariate channel is divided by one factor:

            - ``'spectral'`` (default): the channel's average spectral norm, which
              equalizes covariate scales across predictor variables.
            - ``'l1'``/``'l2'``: the predictor's mean absolute deviation or standard
              deviation.
            - ``None``: leave the covariates on their raw scale, without centering.

            The centering and the ``'l1'``/``'l2'`` factors describe the predictor
            and are measured on all of its samples; the ``'spectral'`` norm
            describes the constructed covariates and is measured on the rows that
            ``pad_stim`` retains.

            Prepare data for prediction with ``scale=None`` and apply the fitted
            model's normalization with :meth:`normalize`.
        stim_is_single
            Whether the original stimulus input was a single predictor per segment.
        basis_std
            Standard deviation of the Gaussian basis functions in seconds.
        in_place
            If ``False`` (default), a copy of ``meg`` is made before it is rescaled.
            Set to ``True`` to modify it in place. ``stim`` is never modified.
        pad_stim
            If ``False`` (default), keep only rows whose full lag window is inside
            the stimulus time axis. If ``True``, retain edge rows with zero-padded
            covariates; with normalization those rows then correspond to a raw
            stimulus of 0 (rather than 0 after centering).
        """
        if not meg:
            raise ValueError("meg is empty")
        elif len(meg) != len(stim):
            raise ValueError("meg and stim have different lengths")
        elif scale is not None and scale not in SCALES:
            raise ValueError(f"{scale=}, need None or one of {SCALES}")

        # The design is fully determined by the first segment's predictors
        sensor_dim = meg[0].get_dim('sensor')
        first_time: UTS = meg[0].get_dim('time')
        tstep = first_time.tstep
        trial_length = len(first_time)
        design = TRFDesign.from_stim(stim[0], tstep, tstart, tstop, basis_stride, basis_std, stim_is_single)

        row_slice = slice(None)
        if not pad_stim:
            # covariate_from_stim() fills the full MEG axis with zero-padded lag
            # histories. ``row_slice`` keeps only samples whose complete lag
            # window lies inside the stimulus.
            drop_start = max(0, *design.stop_samples)
            drop_stop = max(0, *(-s for s in design.start_samples))
            row_slice = slice(drop_start, -drop_stop if drop_stop else None)
        if not len(range(trial_length)[row_slice]):
            raise ValueError(f"{meg=}: no samples remain after applying lag-validity crop")

        # Validate all segments before building anything, so that invalid input
        # fails before any work is done, and before in_place rescales any of the
        # caller's arrays
        ys = []
        for i_segment, (m, ss) in enumerate(zip(meg, stim)):
            meg_time: UTS = m.get_dim('time')
            if m.get_dim('sensor') != sensor_dim:
                raise ValueError(f'{meg=}: combining data segments with different sensor configurations is not supported')
            elif meg_time.tstep != tstep:
                raise ValueError(f"{meg=}: segment {i_segment} time-step incompatible with first segment")
            elif len(meg_time) != trial_length:
                raise NotImplementedError(f"{meg=}: unequal trial length")
            elif stim_dimensions(ss) != design.stim_dims:
                raise ValueError(f"{stim=}: segment {i_segment} dimensions incompatible with first segment")
            for x in ss:
                if x.get_dim('time') != meg_time:
                    raise ValueError(f"segment {i_segment} stim {x!r}: time axis incompatible with meg")
            y = m.get_data(('sensor', 'time'))[:, row_slice]
            flat = np.var(y, axis=1) == 0
            if flat.any():
                raise ValueError(f"{meg=}: segment {i_segment} has flat channels ({', '.join(sensor_dim.names[flat])})")
            ys.append(y)
        if scale is not None:
            # also rejects predictors that are constant over time
            baseline, stim_scaling = get_scaling(stim, design, scale)

        # Filter length/start per expanded covariate channel
        fl_rep = np.repeat(design.filter_length, design.stim_lens)
        st_rep = np.repeat(design.start_samples, design.stim_lens)

        meg_arrays: list[FloatArray] = []
        covariate_arrays: list[FloatArray] = []
        for m, y, ss in zip(meg, ys, stim):
            # Normalize MEG array
            y = y.astype(np.float64, copy=False)
            if not in_place and np.shares_memory(y, m.x):
                y = y.copy()
            norm_factor = sqrt(y.shape[1])
            y /= norm_factor
            meg_arrays.append(y)

            # Build basis-projected covariate matrix
            raw_covs = [x[row_slice] for x in covariate_from_stim(ss, fl_rep, st_rep)]
            covariate_arrays.append(_project_basis(raw_covs, design, norm_factor))

        data = cls(meg_arrays, covariate_arrays, design, sensor_dim)

        if scale is not None:
            # the covariate arrays were constructed above and are not shared, so
            # normalization can modify them without copying
            if stim_scaling is None:
                # 'spectral': measured on the centered covariates, so center first
                data = data.normalize(replace(design, stim_baseline=baseline), copy=False)
                design, stim_scaling = data.design, data._spectral_norms()
            data = data.normalize(replace(design, stim_baseline=baseline, stim_scaling=stim_scaling, scale=scale), copy=False)
        return data

    def __iter__(self) -> Iterator[TrialData]:
        return zip(self.meg, self.covariates)

    def __len__(self) -> int:
        return len(self.meg)

    def __repr__(self) -> str:
        n_segments = len(self.meg)
        if self.meg:
            n_sensors, n_samples = self.meg[0].shape
        else:
            n_sensors, n_samples = len(self.sensor_dim), 0
        n_covariates = self.covariates[0].shape[1] if self.covariates else 0
        predictors = tuple(self.design.stim_names)
        whitened = self.is_whitened
        return f"<{type(self).__name__}: {_count_repr(n_segments, 'segment')}, {_count_repr(n_sensors, 'sensor')}, {_count_repr(n_samples, 'sample')}/segment, {_count_repr(n_covariates, 'covariate')}, {predictors=}, {whitened=}>"

    def __getstate__(self) -> dict[str, Any]:
        return pickle_state(self)

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

    def _spectral_norms(self) -> FloatArray:
        """Spectral norm of each covariate channel, averaged across segments."""
        splits = np.cumsum(self.design.basis_widths)[:-1]
        norms = [[linalg.norm(block, 2) for block in np.split(cov, splits, axis=1)] for cov in self.covariates]
        return np.array(norms).mean(axis=0)

    def normalize(self, design: TRFDesign, copy: bool = True) -> RegressionData:
        """Return a dataset carrying the centering and scaling recorded in ``design``.

        Normalization is a linear operation on the covariates, so for rows with a
        full lag window applying it here is equivalent to applying it to the
        stimulus before covariate construction. Zero-padded edge rows retained
        with ``pad_stim=True`` are the exception: they keep representing a raw
        stimulus of 0, whereas centering the stimulus first would make its
        padding represent the mean.
        Use this to prepare data for a fitted model, which can only be applied to
        covariates on the scale it was fit on::

            data = data.normalize(model.design)

        Parameters
        ----------
        design
            Design specifying the normalization to apply; it must describe the same
            coefficient space as this dataset's design. Steps this dataset already
            carries are skipped, so normalizing twice is a no-op.
        copy
            Copy the covariates before modifying them (default). ``False`` is
            reserved for covariate arrays no other dataset shares, such as the
            ones :meth:`from_data` has just constructed.

        Notes
        -----
        By default the covariates are never modified in place. :meth:`whiten`
        hands out a dataset that shares covariate arrays with this one, and
        writing through them would leave the other dataset carrying a
        normalization that its own ``design`` does not record.

        Raises
        ------
        ValueError
            If ``design`` describes a different coefficient space, a different
            normalization than the covariates already carry, or a scaling factor
            that is not finite and strictly positive.
        """
        self.design.assert_compatible(design)
        baseline, scaling = self.design.normalization_to(design)
        if baseline is None and scaling is None:
            return replace(self, design=design)

        covariates = [cov.copy() for cov in self.covariates] if copy else list(self.covariates)
        if baseline is not None:
            # For a row with a full lag window, subtracting a constant from the
            # stimulus offsets each covariate column by a constant. The offset is
            # applied to every row, so zero-padded edge rows (pad_stim=True)
            # represent a raw stimulus of 0 rather than 0 after centering.
            offset = design.expand(baseline) * design.basis_column_sums / self.norm_factor
            for cov in covariates:
                cov -= offset
        if scaling is not None:
            _check_scaling(scaling, design)
            factors = design.expand(scaling)
            for cov in covariates:
                cov /= factors
        return replace(self, covariates=covariates, design=design)

    def whiten(self, whitening_filter: FloatArray) -> RegressionData:
        """Return a dataset with MEG whitened.

        Parameters
        ----------
        whitening_filter
            Whitening matrix. If the dataset was already whitened with this same
            filter, it is returned unchanged.

        Notes
        -----
        Uses shallow copies of unmodified data.

        Raises
        ------
        ValueError
            If the dataset is already whitened with a different filter: its
            sensor space is not the one this filter belongs to, and whitening
            twice is not equivalent to whitening once with the second filter
            (``W₂ @ W₁ @ meg ≠ W₂ @ meg``).
        """
        if self.whitener is not None:
            if self.whitener is whitening_filter or (self.whitener.shape == whitening_filter.shape and np.allclose(self.whitener, whitening_filter)):
                return self
            raise ValueError("data is already whitened with a different filter, so it belongs to a different (forward model's) sensor space; rebuild the dataset from raw data")
        meg = [np.dot(whitening_filter, m) for m in self.meg]
        return replace(self, meg=meg, whitener=whitening_filter)

    def timeslice(self, idx: Sequence[int] | IndexArray | npt.NDArray[np.bool_]) -> RegressionData:
        """Return a new dataset restricted to selected time indices.

        If this dataset ``.is_whitened``, the returned dataset is also
        marked as whitened and quadratic forms are recomputed lazily.

        Parameters
        ----------
        idx
            Time samples to retain, as integer indices or a boolean mask.
        """
        idx = np.asarray(idx)
        if idx.dtype == bool:
            # a mask's len() is the full axis, not the number of retained samples
            idx = np.flatnonzero(idx)
        mul = self.norm_factor / sqrt(len(idx))
        meg = [m[:, idx] * mul for m in self.meg]
        covariates = [c[idx, :] * mul for c in self.covariates]
        return replace(self, meg=meg, covariates=covariates)
