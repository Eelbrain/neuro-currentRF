"""Regression dataset and covariate construction for NCRF fitting.

``RegressionData`` holds Eelbrain objects and the ``TRFDesign`` describing how
they map onto the regression, and derives the numeric arrays the solver consumes
from them on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from math import sqrt
from typing import Any, TYPE_CHECKING
from collections.abc import Iterator, Sequence

from eelbrain import NDVar, Sensor, UTS
import numpy as np
import numpy.typing as npt
from scipy import linalg

from ._trf_design import TRFDesign, stim_dimensions
from ._forward import _assert_sensors_equal
from ._pickle import pickle_state
from ._repr import _count_repr
from ._typing import FloatArray, IndexArray, Scale, ScaleArg, TrialData

if TYPE_CHECKING:
    from ._forward import ForwardModel


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
    _check_not_constant(by_predictor, design)

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


def _check_not_constant(
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
        stims: Sequence[NDVar],
        Ms: Sequence[int] | npt.ArrayLike,
        starts: Sequence[int] | npt.ArrayLike,
) -> list[FloatArray]:
    """Form lagged covariate matrices from one or more stimulus NDVars.

    Parameters
    ----------
    stims
        Predictor variables. Each predictor must provide a ``time`` axis and
        may have at most one additional feature dimension.
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
        samples: IndexArray,
        norm_factor: float,
) -> FloatArray:
    """Project the retained samples of each channel's lag matrix onto its predictor's Gaussian basis.

    Parameters
    ----------
    raw_covs
        Lag matrix of each expanded covariate channel, in design order.
    design
        Design supplying the basis of each predictor.
    samples
        Rows of the lag matrices to retain.
    norm_factor
        Value the covariates are divided by, matching the MEG data.

    Returns
    -------
    ndarray
        Covariate matrix, shape ``(n_samples, n_basis_cols)``.
    """
    covariates = []
    i = 0
    for n, basis in zip(design.stim_lens, design.basis):
        covariates.extend(np.dot(x[samples], basis) / norm_factor for x in raw_covs[i:i + n])
        i += n
    return np.concatenate(covariates, axis=1).astype(np.float64, copy=False)


@dataclass(eq=False, repr=False)
class RegressionData:
    """Dataset for NCRF fitting: M/EEG segments, their predictors, and the design.

    Use :meth:`from_data` to derive the design, including its normalization, from
    the data itself.

    .. warning::
       The ``meg`` and ``stim`` NDVars are referenced rather than copied, and the
       arrays are built from them when first accessed. Do not modify the NDVars in
       place while the dataset, or one derived from it, is in use: arrays built
       afterwards would reflect the change, but the normalization recorded on the
       design would not.

    Parameters
    ----------
    meg
        M/EEG segments, each an NDVar with ``sensor`` and ``time`` dimensions. All
        segments share the sensor dimension and the time axis length and step.
    stim
        Predictors, one list per segment with one NDVar per predictor variable.
        Each predictor may be 1-D over time or carry one feature dimension before
        time, and shares the time axis of its segment's ``meg``.
    design
        Design describing the predictors, the TRF lags and basis, and the
        normalization to apply to the covariates.
    samples
        Indices of the time samples the dataset covers, into the time axis of
        ``meg``; ``None`` for all samples. :meth:`from_data` drops samples whose
        lag window extends beyond the stimulus, and :meth:`timeslice` selects
        cross-validation folds.
    forward
        Forward model whose whitening filter is applied to the responses, or
        ``None`` for raw data. Set by :meth:`whiten`; whitened data thus carries
        the forward model it is fit with, and recording it lets :meth:`whiten`
        distinguish a no-op (same filter) from an error (different filter).

    Notes
    -----
    The stored fields are the raw inputs. The arrays the solver consumes,
    :attr:`responses` and :attr:`covariates`, are derived from them on demand
    according to :attr:`design`, which fixes the TRF lags, the Gaussian basis the
    predictors are projected onto, and the centering and scaling of the
    covariates.
    """

    meg: list[NDVar]
    stim: list[Sequence[NDVar]]
    design: TRFDesign
    samples: IndexArray | None = None
    forward: ForwardModel | None = None

    def __post_init__(self) -> None:
        if not self.meg:
            raise ValueError("meg is empty")
        elif len(self.meg) != len(self.stim):
            raise ValueError(f"{_count_repr(len(self.meg), 'MEG segment')} but {_count_repr(len(self.stim), 'stimulus list')}")
        sensor_dim = self.sensor_dim
        if self.forward is not None:
            _assert_sensors_equal(sensor_dim.names, self.forward.sensor.names, 'data', 'forward model')
        time: UTS = self.meg[0].get_dim('time')
        if time.tstep != self.design.tstep:
            raise ValueError(f"meg time step {time.tstep} does not match design.tstep {self.design.tstep}")
        for i, (m, ss) in enumerate(zip(self.meg, self.stim)):
            meg_time: UTS = m.get_dim('time')
            if m.get_dim('sensor') != sensor_dim:
                raise ValueError(f"segment {i}: combining data segments with different sensor configurations is not supported")
            elif meg_time.tstep != time.tstep:
                raise ValueError(f"segment {i}: meg time step incompatible with first segment")
            elif len(meg_time) != len(time):
                raise NotImplementedError(f"segment {i}: unequal trial length")
            elif stim_dimensions(ss) != self.design.stim_dims:
                raise ValueError(f"segment {i}: stim dimensions {stim_dimensions(ss)} do not match design.stim_dims {self.design.stim_dims}")
            for x in ss:
                if x.get_dim('time') != meg_time:
                    raise ValueError(f"segment {i} stim {x!r}: time axis incompatible with meg")
        if self.samples is None:
            self.samples = np.arange(len(time))
        else:
            samples = np.asarray(self.samples)
            if samples.dtype == bool:
                samples = np.flatnonzero(samples)
            if not len(samples):
                raise ValueError("samples is empty")
            elif samples.min() < 0 or samples.max() >= len(time):
                raise ValueError(f"samples out of range for {_count_repr(len(time), 'time sample')}")
            self.samples = samples
        if self.design.stim_scaling is not None:
            _check_scaling(self.design.stim_scaling, self.design)

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
            pad_stim: bool = False,
    ) -> RegressionData:
        """Construct a dataset from MEG and stimulus NDVars, deriving the design from them.

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
            Normalization derived from the data and applied to the covariates:
            ``'spectral'``, ``'l1'`` or ``'l2'`` (see :meth:`normalize`), or
            ``None`` to leave the covariates on their raw scale, without
            centering. To apply a fitted model instead, bring the data onto the
            model's scale with ``data.normalize(model.design)``.
        stim_is_single
            Whether the original stimulus input was a single predictor per segment.
        basis_std
            Standard deviation of the Gaussian basis functions in seconds.
        pad_stim
            If ``False`` (default), keep only samples whose full lag window is
            inside the stimulus time axis. If ``True``, retain edge samples with
            zero-padded covariates; with normalization those samples then
            correspond to a raw stimulus of 0 (rather than 0 after centering).
        """
        if not meg:
            raise ValueError("meg is empty")
        elif len(meg) != len(stim):
            raise ValueError("meg and stim have different lengths")

        # The design is fully determined by the first segment's predictors
        time: UTS = meg[0].get_dim('time')
        design = TRFDesign.from_stim(stim[0], time.tstep, tstart, tstop, basis_stride, basis_std, stim_is_single)

        samples = None
        if not pad_stim:
            # covariate_from_stim() fills the full time axis with zero-padded lag
            # histories; keep only samples whose complete lag window lies inside
            # the stimulus
            drop_start = max(0, *design.stop_samples)
            drop_stop = max(0, *(-s for s in design.start_samples))
            samples = np.arange(drop_start, len(time) - drop_stop)
            if not len(samples):
                raise ValueError(f"{meg=}: no samples remain after applying lag-validity crop")

        data = cls(meg, stim, design, samples)
        # Read the NDVars directly rather than through data.responses, so that the
        # returned dataset does not keep a copy of the unwhitened arrays
        for i, m in enumerate(meg):
            flat = np.var(m.get_data(('sensor', 'time'))[:, data.samples], axis=1) == 0
            if flat.any():
                raise ValueError(f"{meg=}: segment {i} has flat channels ({', '.join(data.sensor_dim.names[flat])})")
        if scale is not None:
            data = data.normalize(scale)
        return data

    @property
    def sensor_dim(self) -> Sensor:
        """Sensor dimension shared by all MEG segments."""
        return self.meg[0].get_dim('sensor')

    @property
    def norm_factor(self) -> float:
        """``sqrt(n_samples)`` that responses and covariates are divided by."""
        return sqrt(len(self.samples))

    @property
    def is_whitened(self) -> bool:
        """Whether the responses are transformed by a whitening filter (see ``forward``)."""
        return self.forward is not None

    @cached_property
    def responses(self) -> list[FloatArray]:
        """M/EEG arrays for the solver, one per segment, each shaped ``(n_sensors, n_samples)``.

        The retained samples of ``meg``, whitened if a ``forward`` model is set, and
        divided by :attr:`norm_factor`.
        """
        responses = []
        for m in self.meg:
            y = m.get_data(('sensor', 'time'))[:, self.samples].astype(np.float64, copy=False) / self.norm_factor
            if self.forward is not None:
                y = np.dot(self.forward.whitening_filter, y)
            responses.append(y)
        return responses

    @cached_property
    def covariates(self) -> list[FloatArray]:
        """Covariate matrices for the solver, one per segment, each shaped ``(n_samples, n_coefficients)``.

        Each predictor is expanded into its lag matrix, projected onto the design's
        basis, and divided by :attr:`norm_factor`; the covariates then receive the
        centering and scaling the design records.
        """
        design = self.design
        filter_lengths = np.repeat(design.filter_length, design.stim_lens)
        starts = np.repeat(design.start_samples, design.stim_lens)
        offset = factors = None
        if design.stim_baseline is not None:
            # For a sample with a full lag window, subtracting a constant from the
            # stimulus offsets each covariate column by a constant. The offset is
            # applied to every sample, so zero-padded edge samples (pad_stim=True)
            # represent a raw stimulus of 0 rather than 0 after centering.
            offset = design.expand(design.stim_baseline) * design.basis_column_sums / self.norm_factor
        if design.stim_scaling is not None:
            factors = design.expand(design.stim_scaling)
        covariates = []
        for ss in self.stim:
            cov = _project_basis(covariate_from_stim(ss, filter_lengths, starts), design, self.samples, self.norm_factor)
            if offset is not None:
                cov -= offset
            if factors is not None:
                cov /= factors
            covariates.append(cov)
        return covariates

    def __iter__(self) -> Iterator[TrialData]:
        return zip(self.responses, self.covariates)

    def __len__(self) -> int:
        return len(self.meg)

    def __repr__(self) -> str:
        n_segments = len(self.meg)
        n_sensors = len(self.sensor_dim)
        n_samples = len(self.samples)
        n_covariates = self.design.n_coefficients
        predictors = tuple(self.design.stim_names)
        whitened = self.is_whitened
        return f"<{type(self).__name__}: {_count_repr(n_segments, 'segment')}, {_count_repr(n_sensors, 'sensor')}, {_count_repr(n_samples, 'sample')}/segment, {_count_repr(n_covariates, 'covariate')}, {predictors=}, {whitened=}>"

    def __getstate__(self) -> dict[str, Any]:
        return pickle_state(self)

    @cached_property
    def bbt(self) -> list[FloatArray]:
        """Per-segment ``B @ B.T`` matrices of the responses."""
        return [np.dot(b, b.T) for b in self.responses]

    @cached_property
    def bE(self) -> list[FloatArray]:
        """Per-segment ``B @ E`` cross-product matrices."""
        return [np.dot(b, E) for b, E in zip(self.responses, self.covariates)]

    @cached_property
    def EtE(self) -> list[FloatArray]:
        """Per-segment ``E.T @ E`` covariate Gram matrices."""
        return [np.dot(E.T, E) for E in self.covariates]

    def _spectral_norms(self) -> FloatArray:
        """Spectral norm of each covariate channel, averaged across segments."""
        splits = np.cumsum(self.design.basis_widths)[:-1]
        norms = [[linalg.norm(block, 2) for block in np.split(cov, splits, axis=1)] for cov in self.covariates]
        return np.array(norms).mean(axis=0)

    def normalize(self, design: TRFDesign | Scale) -> RegressionData:
        """Return a dataset whose covariates are centered and scaled.

        The normalization is either derived from this dataset, for data a model is
        fit on, or taken from a design, for data a fitted model is applied to. A
        fitted model can only be applied to covariates on the scale it was fit
        on::

            data = data.normalize(model.design)

        The covariates are derived from the raw predictors, so any normalization
        this dataset already carries is simply replaced. Normalization is a linear
        operation on the covariates, so for samples with a full lag window it is
        equivalent to normalizing the stimulus before covariate construction.
        Zero-padded edge samples retained with ``pad_stim=True`` are the
        exception: they keep representing a raw stimulus of 0, whereas centering
        the stimulus first would make its padding represent the mean.

        Parameters
        ----------
        design
            A :class:`~ncrf.TRFDesign` whose centering and scaling to apply; it
            must describe the same coefficient space as this dataset's design.
            Alternatively, a scale to derive from this dataset. Each predictor's
            mean is then subtracted, and each covariate channel is divided by one
            factor:

            - ``'spectral'``: the channel's average spectral norm, which
              equalizes covariate scales across predictor variables.
            - ``'l1'``/``'l2'``: the predictor's mean absolute deviation or standard
              deviation.

            The centering and the ``'l1'``/``'l2'`` factors describe the predictor
            and are measured on the whole ``stim`` NDVars, regardless of
            ``samples``; for a dataset restricted with :meth:`timeslice`, they
            therefore include the samples outside the slice. The ``'spectral'``
            norm describes the constructed covariates and is measured on the
            retained ``samples``.

        Raises
        ------
        ValueError
            If ``design`` describes a different coefficient space, or a scaling
            factor that is not finite and strictly positive; when deriving a scale,
            also if a predictor is constant over time.
        """
        if isinstance(design, TRFDesign):
            self.design.assert_compatible(design)
            return replace(self, design=design)
        scale = design
        if scale not in SCALES:
            raise ValueError(f"{scale=}, need one of {SCALES} or a TRFDesign")

        # also rejects predictors that are constant over time
        baseline, stim_scaling = get_scaling(self.stim, self.design, scale)
        centered = replace(self, design=replace(self.design, stim_baseline=baseline, stim_scaling=None, scale=None))
        if stim_scaling is None:
            # 'spectral' is measured on the centered covariates
            stim_scaling = centered._spectral_norms()
        data = replace(centered, design=replace(centered.design, stim_scaling=stim_scaling, scale=scale))
        if scale == 'spectral':
            # The covariates are a function of the design, and centered is discarded,
            # so its covariates can be rescaled in place and handed over rather than
            # rebuilt from the predictors
            covariates = centered.__dict__.pop('covariates')
            factors = data.design.expand(stim_scaling)
            for cov in covariates:
                cov /= factors
            data.__dict__['covariates'] = covariates
        return data

    def whiten(self, forward: ForwardModel) -> RegressionData:
        """Return a dataset whose responses are whitened with ``forward``.

        Parameters
        ----------
        forward
            Forward model for exactly this dataset's sensors, in the same order.
            Its whitening filter is applied to the responses, and it is recorded
            as :attr:`forward`. If the dataset is already whitened with an
            equivalent forward model (same lead field and noise covariance), it
            is returned unchanged.

        Raises
        ------
        ValueError
            If ``forward`` has different sensors than the dataset, or if the
            dataset is already whitened with a different forward model: the
            forward model whitened data carries is the one it is fit with, and
            whitening twice is not equivalent to whitening once with the second
            filter (``W₂ @ W₁ @ meg ≠ W₂ @ meg``).
        """
        _assert_sensors_equal(self.sensor_dim.names, forward.sensor.names, 'data', 'forward model')
        if self.forward is not None:
            current = self.forward
            if current is forward or (current.lead_field.shape == forward.lead_field.shape and np.allclose(current.lead_field, forward.lead_field) and np.allclose(current.noise_covariance, forward.noise_covariance)):
                return self
            raise ValueError("data is already whitened with a different forward model; fitting or evaluating it with this one would use the wrong lead field or whitening, so rebuild the dataset from raw data")
        data = replace(self, forward=forward)
        # The covariates do not depend on the whitening, so share them
        for key in ('covariates', 'EtE'):
            if key in self.__dict__:
                data.__dict__[key] = self.__dict__[key]
        return data

    def timeslice(self, idx: Sequence[int] | IndexArray | npt.NDArray[np.bool_]) -> RegressionData:
        """Return a new dataset restricted to a subset of this dataset's samples.

        Parameters
        ----------
        idx
            Samples to retain, as integer indices or a boolean mask into this
            dataset's ``samples``.
        """
        return replace(self, samples=self.samples[np.asarray(idx)])
