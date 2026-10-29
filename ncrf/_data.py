"""Regression dataset for NCRF fitting.

``RegressionData`` combines Eelbrain's deconvolution dataset (the M/EEG and
predictor segments and their normalization) with the ``TRFDesign`` describing
how the predictors map onto the regression, and derives the numeric arrays the
solver consumes from them on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from math import sqrt
from typing import Any, get_args
from collections.abc import Iterator, Sequence

from eelbrain import Dataset, Sensor
from eelbrain.trf import DeconvolutionData, lag_matrix
import numpy as np
import numpy.typing as npt
from scipy import linalg

from ._trf_design import TRFDesign
from ._forward import ForwardModel, _assert_sensors_equal
from ._pickle import pickle_state
from ._repr import _count_repr
from ._typing import FloatArray, NDVarArg, Scale, ScaleArg, TrialData


SCALES: tuple[Scale, ...] = get_args(Scale)
#: A contiguous part of a data segment: ``(i_segment, start, stop)``, with
#: ``start`` and ``stop`` as sample indices into the segment.
Block = tuple[int, int, int]


def _check_scaling(
        scaling: FloatArray,
        scale: ScaleArg,
        names: Sequence[str],
) -> None:
    """Check that scaling factors can be divided by without destroying the covariates.

    Dividing by a factor of 0 fills the covariate columns with NaN, and a negative
    or non-finite factor corrupts them just as silently, so a design carrying such
    factors must be rejected rather than applied.

    Parameters
    ----------
    scaling
        Scaling factor of each covariate channel.
    scale
        Which scaling the factors stem from, for the error message.
    names
        Name of each covariate channel, for the error message.

    Raises
    ------
    ValueError
        If any factor is not finite and strictly positive.
    """
    bad = ~(np.isfinite(scaling) & (scaling > 0))
    if not bad.any():
        return
    items = ', '.join(f'{names[i]}={scaling[i]:g}' for i in np.flatnonzero(bad))
    raise ValueError(f"invalid {scale!r} scaling ({items}): scaling factors must be finite and > 0; a predictor that is constant over time has no variation to scale by")


def _default_blocks(
        data: DeconvolutionData,
        design: TRFDesign,
        pad_stim: bool,
) -> list[Block]:
    """The samples of each segment that a dataset covers by default.

    Parameters
    ----------
    data
        Segmented data.
    design
        Design whose lag windows determine which samples are valid.
    pad_stim
        Keep all samples (see :meth:`RegressionData.from_data`).

    Raises
    ------
    ValueError
        If no samples remain.
    """
    if pad_stim:
        drop_start = drop_stop = 0
    else:
        # lag_matrix() fills the full time axis with padded lag histories; keep
        # only samples whose complete lag window lies inside the stimulus
        drop_start = max(0, *design.stop_samples)
        drop_stop = max(0, *(-s for s in design.start_samples))
    blocks = [(i, drop_start, n - drop_stop) for i, n in enumerate(data.segment_n_times) if drop_start < n - drop_stop]
    if not blocks:
        raise ValueError("no samples remain after applying lag-validity crop; the segments are shorter than the TRF")
    return blocks


@dataclass(eq=False, repr=False)
class RegressionData:
    """Dataset for NCRF fitting: M/EEG segments, their predictors, and the design.

    Use :meth:`from_data` to construct the dataset from NDVars and derive the
    design, including its normalization, from the data itself.

    Parameters
    ----------
    data
        M/EEG segments and their predictors. The M/EEG data needs ``sensor`` and
        ``time`` dimensions; each predictor is 1-D over time or carries feature
        dimensions before time. The predictors are normalized in place by
        :meth:`normalize`.
    design
        Design describing the predictors, the TRF lags and basis, and the
        normalization applied to the predictors.
    blocks
        The samples the dataset covers, as ``(i_segment, start, stop)`` for each
        contiguous part. By default, each segment without the samples whose lag
        window extends beyond the stimulus. :meth:`subset` restricts the blocks,
        e.g. to cross-validation folds.
    forward
        Forward model whose whitening filter is applied to the responses, or
        ``None`` for raw data. Set by :meth:`whiten`; whitened data thus carries
        the forward model it is fit with, and recording it lets :meth:`whiten`
        distinguish a no-op (same filter) from an error (different filter).

    Notes
    -----
    The arrays the solver consumes, :attr:`responses` and :attr:`covariates`,
    are derived from the segments on demand according to :attr:`design`, which
    fixes the TRF lags and the Gaussian basis the predictors are projected onto.
    Both are normalized by :attr:`norm_factor`: the responses are divided by it,
    whereas the covariates are views into the per-segment covariate matrices
    (shared by all datasets derived from this one through :meth:`subset` and
    :meth:`whiten`), and the factor is applied in the derived Gram matrices
    (:attr:`bE`, :attr:`EtE`) and in predictions instead.
    """

    data: DeconvolutionData
    design: TRFDesign
    blocks: list[Block] | None = None
    forward: ForwardModel | None = None

    def __post_init__(self) -> None:
        data, design = self.data, self.design
        if data.y is None:
            raise ValueError("data has no M/EEG data (DeconvolutionData without y)")
        ydims = data.y.blocks[0].dims
        if len(ydims) != 1 or not isinstance(ydims[0], Sensor):
            raise ValueError(f"M/EEG data needs a sensor and a time dimension, but has {', '.join(dim.name for dim in ydims)} and time")
        if self.forward is not None:
            _assert_sensors_equal(ydims[0].names, self.forward.sensor.names, 'data', 'forward model')
        if data.time.tstep != design.tstep:
            raise ValueError(f"data time step {data.time.tstep} does not match design.tstep {design.tstep}")
        if data.x.blocks != design.blocks:
            raise ValueError(f"predictors {data.x.names} do not match the design's {[name for block in design.blocks for name in block.names()]}")
        if self.blocks is None:
            self.blocks = _default_blocks(data, design, False)
        else:
            n_times = data.segment_n_times
            for i, start, stop in self.blocks:
                if not 0 <= i < len(n_times):
                    raise ValueError(f"block ({i}, {start}, {stop}): segment index out of range for {_count_repr(len(n_times), 'segment')}")
                elif not 0 <= start < stop <= n_times[i]:
                    raise ValueError(f"block ({i}, {start}, {stop}): out of range for segment with {_count_repr(n_times[i], 'sample')}")
        if design.stim_scaling is not None:
            _check_scaling(design.stim_scaling, design.scale, data.x.names)

    @classmethod
    def from_data(
            cls,
            y: NDVarArg,
            x: NDVarArg | Sequence[NDVarArg],
            tstart: float | Sequence[float],
            tstop: float | Sequence[float],
            basis_stride: int = 1,
            scale: ScaleArg = 'spectral',
            basis_std: float = 0.0085,
            pad_stim: bool = False,
            data: Dataset | None = None,
    ) -> RegressionData:
        """Construct a dataset from M/EEG and predictor NDVars, deriving the design from them.

        Parameters
        ----------
        y
            M/EEG data with ``sensor`` and ``time`` dimensions. Equal-length
            segments can be packed into an NDVar with a ``case`` dimension,
            unequal-length segments supplied as :class:`~eelbrain.Datalist` of
            NDVars.
        x
            Predictors, matching ``y`` in segments and time axis. One predictor
            variable is given as NDVar (or Datalist), several as a sequence of
            them. Each predictor is 1-D over time or carries feature dimensions
            before time.
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
            Normalization derived from the data and applied to the predictors:
            ``'spectral'``, ``'l1'`` or ``'l2'`` (see :meth:`normalize`), or
            ``None`` to leave the predictors on their raw scale, without
            centering. To apply a fitted model instead, bring the data onto the
            model's scale with ``data.normalize(model.design)``.
        basis_std
            Standard deviation of the Gaussian basis functions in seconds.
        pad_stim
            If ``False`` (default), keep only samples whose full lag window is
            inside the stimulus time axis. If ``True``, retain edge samples with
            padded covariates, in which the stimulus outside its time axis is 0
            (on the raw scale, i.e. before normalization).
        data
            Dataset in which to evaluate ``y`` and ``x`` if they are strings.

        Raises
        ------
        ValueError
            If the M/EEG data has a flat channel or contains NaN, which would
            corrupt the whitened data and the fit metrics.
        """
        if scale is not None and scale not in SCALES:
            raise ValueError(f"{scale=}, need None or one of {SCALES}")
        dec_data = DeconvolutionData(x, y, data)
        dec_data.y.check()
        design = TRFDesign.from_data(dec_data, tstart, tstop, basis_stride, basis_std)
        out = cls(dec_data, design, _default_blocks(dec_data, design, pad_stim))
        if scale is not None:
            out = out.normalize(scale)
        return out

    @property
    def sensor_dim(self) -> Sensor:
        """Sensor dimension of the M/EEG data."""
        return self.data.y.blocks[0].dims[0]

    @property
    def n_samples(self) -> int:
        """Number of time samples the dataset covers, across all blocks."""
        return sum(stop - start for _, start, stop in self.blocks)

    @property
    def norm_factor(self) -> float:
        """Square root of the mean number of samples per block, by which responses and covariates are normalized.

        Normalizing by the per-block sample count keeps the objective, and hence
        the meaning of the regularization parameter, independent of the length
        of the data; with blocks of unequal length, each block contributes in
        proportion to its length.
        """
        return sqrt(self.n_samples / len(self.blocks))

    @property
    def is_whitened(self) -> bool:
        """Whether the responses are transformed by a whitening filter (see ``forward``)."""
        return self.forward is not None

    @cached_property
    def responses(self) -> list[FloatArray]:
        """M/EEG arrays for the solver, one per block, each shaped ``(n_sensors, n_samples)``.

        The samples of each block, whitened if a ``forward`` model is set, and
        divided by :attr:`norm_factor`.
        """
        responses = []
        for i, start, stop in self.blocks:
            y = self.data.y.segments[i][:, start:stop] / self.norm_factor
            if self.forward is not None:
                y = np.dot(self.forward.whitening_filter, y)
            responses.append(y)
        return responses

    @cached_property
    def _segment_covariates(self) -> list[FloatArray]:
        """Covariate matrices of the complete data segments, ``(n_times, n_coefficients)``.

        Each predictor channel is expanded into its lag matrix and projected onto
        the design's basis. Outside of the segment, the predictors are padded with
        the (normalized) value of 0. Shared by all datasets derived from this one.
        """
        design = self.design
        pads = self.data.x.pads
        covariates = []
        for x_seg in self.data.x.segments:
            columns = []
            for block, basis, start, n_lags in zip(design.blocks, design.basis, design.start_samples, design.filter_length):
                columns.extend(np.dot(lag_matrix(x, start, n_lags, pad), basis) for x, pad in zip(x_seg[block.index], pads[block.index]))
            covariates.append(np.concatenate(columns, axis=1))
        return covariates

    @property
    def covariates(self) -> list[FloatArray]:
        """Covariate matrices for the solver, one per block, each shaped ``(n_samples, n_coefficients)``.

        Views into the covariate matrices of the complete segments; not divided by
        :attr:`norm_factor` (see the class notes).
        """
        covariates = self._segment_covariates
        return [covariates[i][start:stop] for i, start, stop in self.blocks]

    def __iter__(self) -> Iterator[TrialData]:
        return zip(self.responses, self.covariates)

    def __len__(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        n_segments = len(self.data.segments)
        n_sensors = len(self.sensor_dim)
        n_samples = self.n_samples
        n_covariates = self.design.n_coefficients
        predictors = tuple(self.design.stim_names)
        scale = self.design.scale
        whitened = self.is_whitened
        return f"<{type(self).__name__}: {_count_repr(n_segments, 'segment')}, {_count_repr(n_sensors, 'sensor')}, {_count_repr(n_samples, 'sample')}, {_count_repr(n_covariates, 'covariate')}, {predictors=}, {scale=}, {whitened=}>"

    def __getstate__(self) -> dict[str, Any]:
        return pickle_state(self)

    @cached_property
    def bbt(self) -> list[FloatArray]:
        """Per-block ``B @ B.T`` matrices of the responses."""
        return [np.dot(b, b.T) for b in self.responses]

    @cached_property
    def bE(self) -> list[FloatArray]:
        """Per-block ``B @ E`` cross-product matrices (with ``E`` normalized)."""
        return [np.dot(b, E) / self.norm_factor for b, E in zip(self.responses, self.covariates)]

    @cached_property
    def EtE(self) -> list[FloatArray]:
        """Per-block ``E.T @ E`` covariate Gram matrices (with ``E`` normalized)."""
        return [np.dot(E.T, E) / self.norm_factor ** 2 for E in self.covariates]

    def _spectral_norms(self) -> FloatArray:
        """Spectral norm of each (normalized) covariate channel, averaged across blocks."""
        splits = np.cumsum(self.design.basis_widths)[:-1]
        norms = [[linalg.norm(block, 2) for block in np.split(cov, splits, axis=1)] for cov in self.covariates]
        return np.array(norms).mean(axis=0) / self.norm_factor

    def _derive(self, **changes: Any) -> RegressionData:
        """A dataset with changed fields, taking over this dataset's derived arrays where they remain valid.

        Parameters
        ----------
        **changes
            Field values for the new dataset. The covariates are shared unless
            the design changes, the responses unless the blocks or the forward
            model change.
        """
        out = replace(self, **changes)
        shared = []
        if 'design' not in changes:
            shared.append('_segment_covariates')
            if 'blocks' not in changes:
                shared.append('EtE')
        if 'blocks' not in changes and 'forward' not in changes:
            shared.extend(('responses', 'bbt'))
        for key in shared:
            if key in self.__dict__:
                out.__dict__[key] = self.__dict__[key]
        return out

    def normalize(self, design: TRFDesign | Scale) -> RegressionData:
        """Normalize the predictors in place, and return the dataset with the updated design.

        The normalization is either derived from this dataset, for data a model is
        fit on, or taken from a design, for data a fitted model is applied to. A
        fitted model can only be applied to covariates on the scale it was fit
        on::

            data = data.normalize(model.design)

        Each predictor's mean is subtracted, and each predictor channel is
        divided by one factor. Normalization is a linear operation on the
        predictors, and hence on the covariates built from them. The data
        layer records the values it applied (``data.x.mean``, ``data.x.scale``),
        and the design records them as ``stim_baseline`` and ``stim_scaling``.

        The predictor arrays are modified in place and hence shared with the
        returned dataset; the returned dataset should be used from then on.
        Normalizing a dataset that is already normalized is an error, since the
        normalization recorded on its design describes the predictors on their
        original scale.

        Parameters
        ----------
        design
            A :class:`~ncrf.TRFDesign` whose centering and scaling to apply; it
            must describe the same coefficient space as this dataset's design.
            Alternatively, a scale to derive from this dataset:

            - ``'spectral'``: the channel's average spectral norm, which
              equalizes covariate scales across predictor variables (measured on
              the covariates of the retained samples).
            - ``'l1'``/``'l2'``: the predictor's mean absolute deviation or standard
              deviation (measured on all samples of the predictor).

        Raises
        ------
        ValueError
            If ``design`` describes a different coefficient space, or a scaling
            factor that is not finite and strictly positive; when deriving a scale,
            also if a predictor is constant over time or contains NaN.
        """
        data = self.data
        if isinstance(design, TRFDesign):
            self.design.assert_compatible(design)
            if data.x.mean is not None:
                try:
                    self.design.assert_same_normalization(design)
                except ValueError:
                    raise ValueError("data is already normalized, and can not be normalized with a different design; rebuild the dataset from raw data") from None
                return self
            if design.stim_scaling is not None:
                _check_scaling(design.stim_scaling, design.scale, data.x.names)
            if design.stim_baseline is not None or design.stim_scaling is not None:
                data.x.normalize(False if design.stim_baseline is None else design.stim_baseline, 1. if design.stim_scaling is None else design.stim_scaling)
            return self._derive(design=replace(self.design, stim_baseline=design.stim_baseline, stim_scaling=design.stim_scaling, scale=design.scale))
        scale = design
        if scale not in SCALES:
            raise ValueError(f"{scale=}, need one of {SCALES} or a TRFDesign")
        elif data.x.mean is not None:
            raise ValueError(f"data is already normalized with scale={self.design.scale!r}; rebuild the dataset from raw data to normalize it differently")
        data.x.check()
        if scale == 'spectral':
            data.x.normalize(True, 1.)
            centered = self._derive(design=replace(self.design, stim_baseline=data.x.mean, stim_scaling=None, scale=None))
            scaling = centered._spectral_norms()
            _check_scaling(scaling, scale, data.x.names)
            data.x.normalize(False, scaling)
            # The covariates are linear in the predictors, so the centered ones can
            # be rescaled in place and handed over rather than rebuilt
            covariates = centered.__dict__.pop('_segment_covariates')
            factors = self.design.expand(scaling)
            for cov in covariates:
                cov /= factors
            out = centered._derive(design=replace(self.design, stim_baseline=data.x.mean, stim_scaling=data.x.scale, scale=scale))
            out.__dict__['_segment_covariates'] = covariates
            return out
        data.x.normalize(True, scale)
        return self._derive(design=replace(self.design, stim_baseline=data.x.mean, stim_scaling=data.x.scale, scale=scale))

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
        return self._derive(forward=forward)

    def subset(self, ranges: npt.ArrayLike) -> RegressionData:
        """Return a dataset restricted to a subset of this dataset's samples.

        Parameters
        ----------
        ranges
            ``(n, 2)`` array of ``[start, stop]`` sample indices on the
            concatenated time axis of the segments, such as
            :attr:`eelbrain.trf.Split.train`. Only samples that this dataset covers
            are retained.

        Notes
        -----
        The subset shares the covariate matrices of the complete segments with
        this dataset, so cross-validation folds cost no more than their
        responses.

        Raises
        ------
        ValueError
            If no samples of this dataset are in ``ranges``.
        """
        blocks = []
        for i, start, stop in self.data.segment_slices(np.asarray(ranges)):
            for j, lo, hi in self.blocks:
                if i == j and max(start, lo) < min(stop, hi):
                    blocks.append((i, max(start, lo), min(stop, hi)))
        if not blocks:
            raise ValueError(f"{ranges=}: contains none of the dataset's samples")
        out = self._derive(blocks=blocks)
        # build the per-segment covariates here, so that all subsets share them
        out.__dict__['_segment_covariates'] = self._segment_covariates
        return out
