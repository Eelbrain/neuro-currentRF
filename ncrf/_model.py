"""The NCRF estimator, reusable fitted model, and fit report.

:class:`NCRFEstimator` owns forward-model preparation, candidate selection, and
fitting. :class:`NCRF` contains the resulting coefficients and prediction API,
independent of the solver that produced them. :class:`NCRFFit` bundles that
model with training scores and solver-specific provenance.
"""
# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Christian Brodbeck <email:brodbecc@mcmaster.ca>
# License: BSD (3-clause)
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any
from collections.abc import Sequence

from eelbrain import NDVar, Sensor, UTS, fmtxt
import numpy as np

from ._crossvalidation import CrossValidation, CVResult
from ._data import RegressionData
from ._trf_design import TRFDesign
from ._forward import ForwardModel, _assert_sensors_equal, _noise_covariance
from ._metrics import Metric, explained_variance, l2_error, merge_scores
from ._pickle import pickle_state
from ._repr import _count_repr, _forward_summary
from ._solvers import Solver, SolverFit
from ._typing import FloatArray, NoiseArg


#: Layout of the NCRF pickle state; bump on incompatible attribute changes.
#: Pre-0.5 pickles carry no stamp and load as version 0.
_PICKLE_VERSION = 1


class NCRF:
    """Fitted NCRF model that can be applied to compatible datasets.

    Holds the estimated weights together with the forward model and stimulus
    design needed to reconstruct response functions, predict whitened sensor
    data, and evaluate predictions. Reusable and picklable; produced by
    :meth:`NCRFEstimator.fit` and exposed as :attr:`NCRFFit.model`.

    Attributes
    ----------
    forward
        The internal forward-model state (lead field, whitening filter, and
        source, sensor, and orientation dimensions).
    theta
        Fitted NCRF coefficients over the Gaussian basis, shape
        ``(source, stimulus * lag)``: rows are the source dipoles in
        lead-field order (orientation varying fastest), columns are
        the predictors of :attr:`design` concatenated, each ordered by stimulus
        with lag varying fastest.
    design
        Stimulus, basis, and normalization metadata of the data the model was fit
        on, including the TRF timing (``design.tstart``, ``design.tstep``,
        ``design.tstop``) and the Gaussian-basis width (``design.basis_std``).
        Pass it to :meth:`~ncrf.RegressionData.normalize` to bring another
        dataset onto the same scale.
    """

    def __init__(
            self,
            forward: ForwardModel,
            theta: FloatArray,
            design: TRFDesign,
    ) -> None:
        self.forward = forward
        self.theta = theta
        self.design = design

    def __repr__(self) -> str:
        n_atoms = self.theta.shape[1]
        predictors = tuple(self.design.stim_names)
        return f"<{type(self).__name__}: {_forward_summary(self.forward)}, {_count_repr(n_atoms, 'basis coefficient')}, {predictors=}>"

    def __getstate__(self) -> dict[str, Any]:
        return {**pickle_state(self), 'version': _PICKLE_VERSION}

    def __setstate__(self, state: dict[str, Any]) -> None:
        # A model pickled with a different attribute layout would unpickle into
        # an unusable object here, since the class path stays the same.
        version = state.pop('version', 0)
        if version != _PICKLE_VERSION:
            raise RuntimeError(f"this model was pickled with an incompatible ncrf version (pickle layout {version}, expected {_PICKLE_VERSION}); re-fit it with the current version")
        self.__dict__.update(state)

    def _theta_for(self, data: RegressionData) -> FloatArray:
        """Coefficients, after checking that ``data`` is on the scale they were fit on.

        ``theta`` is fit against covariates carrying the normalization recorded in
        :attr:`design`, so any other normalization would silently change the
        predictions.
        """
        design = data.design
        if design is self.design:
            return self.theta
        self.design.assert_compatible(design)
        design.normalization_to(self.design, assert_applied=True)
        return self.theta

    @cached_property
    def _whitened_projection(self) -> FloatArray:
        """``whitened_lead_field @ theta``, the segment-independent part of whitened prediction."""
        return np.dot(self.forward.whitened_lead_field, self.theta)

    def _predict_whitened(self, covariate: FloatArray) -> FloatArray:
        """Predicted whitened sensor data for one trial's covariate matrix."""
        return np.dot(self._whitened_projection, covariate.T)

    def predict(
            self,
            data: RegressionData,
            *,
            whitened: bool = False,
    ) -> list[FloatArray]:
        """Predict sensor-space data for each segment.

        Only the covariates are used, so it does not matter whether ``data`` has
        been whitened.

        Parameters
        ----------
        data
            Prepared dataset with a design compatible with the training data, and
            carrying the same normalization (prepare it with ``scale=None`` and
            apply :meth:`~ncrf.RegressionData.normalize` with :attr:`design`).
        whitened
            Predict in the whitened, variance-normalized sensor space the model is
            fit in, i.e. the space :meth:`evaluate` scores in, rather than in the
            units of the original M/EEG data.

        Returns
        -------
        list
            Predicted arrays, one per segment, each shaped
            ``(n_sensors, n_times)``.
        """
        _assert_sensors_equal(data.sensor_dim.names, self.forward.sensor.names, 'data', 'forward model')
        theta = self._theta_for(data)
        if whitened:
            return [self._predict_whitened(covariate) for covariate in data.covariates]
        # Predicting through the un-whitened lead field, and undoing the sqrt(n_times)
        # by which both MEG and covariates were divided, puts the prediction back into
        # the units of the M/EEG data the dataset was built from.
        source = np.dot(self.forward.lead_field, theta) / self.forward.lead_field_scaling
        return [np.dot(source, covariate.T) * data.norm_factor for covariate in data.covariates]

    def evaluate(
            self,
            data: RegressionData,
            metrics: Sequence[Metric] = (explained_variance, l2_error),
    ) -> dict[str, float]:
        """Score predictions on ``data`` with one or more metrics.

        The data is whitened and predicted once, and every metric is evaluated on
        those predictions. Scoring happens in whitened sensor space, where the noise
        is isotropic and channels are therefore comparable; this is the space the
        solver optimizes in and the one cross-validation compares candidates in. Use
        ``predict(data, whitened=True)`` to obtain the predictions the metrics see.

        Parameters
        ----------
        data
            Prepared dataset with a design compatible with the training data, and
            carrying the same normalization (see :meth:`predict`).
        metrics
            Metric functions such as :func:`~ncrf.explained_variance`, each
            mapping observed and predicted per-segment arrays to a scalar.
            Results are keyed by function name.
        """
        data = self.forward.whiten(data)
        self._theta_for(data)
        observed = [meg for meg, _ in data]
        predicted = [self._predict_whitened(covariate) for _, covariate in data]
        return {metric.__name__: metric(observed, predicted) for metric in metrics}

    def voxelwise_explained_variance(self, data: RegressionData) -> NDVar:
        """Compute each source's contribution to explained variance.

        ``data`` has to carry the same normalization as the training data (see
        :meth:`predict`).
        """
        data = self.forward.whiten(data)
        theta = self._theta_for(data)
        W_leadfield = self.forward.whitened_lead_field
        temp = np.zeros(len(self.forward.source))
        for meg, covariate in data:
            total_var = np.var(meg, axis=1)
            y_full = meg - self._predict_whitened(covariate)
            base_var = np.var(y_full, axis=1)
            for i in range(len(self.forward.source)):
                # Zeroing source i's weights just removes its (linear) contribution
                # to the prediction, so add that contribution back to the residual.
                block = self.forward.source_block(i)
                contribution = np.dot(np.dot(W_leadfield[:, block], theta[block]), covariate.T)
                y_i = y_full + contribution
                temp[i] += np.nansum((np.var(y_i, axis=1) - base_var) / total_var) / meg.shape[0]

        return NDVar(temp / len(data), self.forward.source)

    @cached_property
    def h(self) -> NDVar | list[NDVar]:
        """The spatio-temporal response function as (list of) :class:`eelbrain.NDVar`.

        Expands the Gabor coefficients in :attr:`theta` back into labeled response
        functions, one per predictor variable (or a bare NDVar when the model was
        fit on a single predictor).

        ``h`` is in source-current units per unit of the *normalized* covariates
        the model was fit on (``scale`` argument). Use :attr:`~ncrf.NCRF.h_scaled`
        for the response in units of the original stimulus.
        """
        design = self.design
        space = self.forward.space
        source_dims = (self.forward.source, space) if space else (self.forward.source,)

        h = []
        start = 0
        for basis, stim_len, dim, name, tstart in zip(design.basis, design.stim_lens, design.stim_dims, design.stim_names, design.tstart):
            # Index corresponding theta; theta shape: (source, stimulus * lag)
            stop = start + basis.shape[1] * stim_len
            x = self.theta[:, start:stop]
            start = stop

            # Reshape to (stimulus, source, lag)
            x = x.reshape((self.theta.shape[0], stim_len, -1)).swapaxes(0, 1)
            x = np.dot(x, basis.T) / self.forward.lead_field_scaling

            # Package in NDVar
            lag_dim = UTS(tstart, design.tstep, x.shape[-1])
            dims = (dim, *source_dims, lag_dim) if dim else (*source_dims, lag_dim)
            h.append(NDVar(x.reshape([len(d) for d in dims]), dims, name=name))

        if design.stim_is_single:
            return h[0]
        return h

    @cached_property
    def h_scaled(self) -> NDVar | list[NDVar]:
        """:attr:`~ncrf.NCRF.h` with the covariate normalization undone.

        In source-current units per unit of the *original* stimulus, and hence
        comparable across predictors and across models fit with different
        ``scale``. Identical to :attr:`~ncrf.NCRF.h` when the covariates were not
        scaled.
        """
        if self.design.stim_scaling is None:
            return self.h
        scaling = self.design.per_predictor(self.design.stim_scaling)
        if self.design.stim_is_single:
            return self.h / scaling[0]
        return [h / s for h, s in zip(self.h, scaling)]


@dataclass(eq=False, repr=False)
class NCRFEstimator:
    """Estimator for neuro-current response functions (NCRFs).

    Construct with :meth:`from_lead_field` from a lead field and noise
    covariance, then call :meth:`fit` with a :class:`RegressionData` instance
    to obtain an :class:`NCRFFit`.

    The estimator stores its inputs in full; the whitened
    :class:`~ncrf.ForwardModel` is derived by :meth:`fit` for exactly the
    channels of the data being fit, since whitening is not separable per
    channel.

    Parameters
    ----------
    lead_field
        Full forward solution with ``sensor`` and ``source`` dimensions and an
        optional ``space`` dimension for free orientation.
    noise_covariance
        Sensor-space noise covariance for :attr:`noise_channels`, in their order.
    noise_channels
        Channel names of ``noise_covariance``, in its order: a subset of the
        lead field's channels. Alignment to the data's channels happens by name
        when the forward model is derived.
    forward
        Whitened forward model, used for data with exactly its channels, in
        order. For other data, :meth:`fit` derives a forward model from
        ``lead_field`` and ``noise_covariance`` without storing it: fitting never
        modifies the estimator (the forward model a fit used is
        ``NCRFFit.model.forward``).

    Notes
    -----
    Use :meth:`RegressionData.from_data` to prepare the M/EEG and predictor
    segments, initialize this estimator with the matching forward inputs, and
    call :meth:`fit` with a configured :class:`Solver`. The returned
    :class:`NCRFFit` keeps the reusable model separate from solver-specific
    fitted state.
    """

    lead_field: NDVar | None = None
    noise_covariance: FloatArray | None = None
    noise_channels: Sequence[str] | None = None
    forward: ForwardModel | None = None

    @classmethod
    def from_lead_field(cls, lead_field: NDVar, noise_covariance: NoiseArg) -> NCRFEstimator:
        """Construct from a lead-field :class:`~eelbrain.NDVar` and noise covariance.

        Parameters
        ----------
        lead_field
            Forward solution a.k.a. lead-field matrix, with ``sensor`` and ``source``
            dimensions and an optional ``space`` dimension for free orientation.
        noise_covariance
            Noise covariance in sensor space, typically estimated from an empty-room
            recording: either directly as :class:`mne.Covariance` or as an
            :class:`eelbrain.NDVar` from which a covariance will be estimated.
            Channels are matched to the lead field by name: the noise channels must
            be a subset of the lead field's channels (channels dropped from the
            noise, e.g. bad channels, are simply excluded from fitting), whereas
            noise for a channel without a lead field is an error.

        Raises
        ------
        ValueError
            If the noise has channels that are not in the lead field. Whitening
            requires a noise estimate for every fitted channel, so noise for a
            channel without a lead field is a sign of misaligned inputs.
        """
        noise_cov, noise_names = _noise_covariance(noise_covariance)
        sensor_names = set(lead_field.get_dim('sensor').names)
        extra = [name for name in noise_names if name not in sensor_names]
        if extra:
            raise ValueError(f"noise covariance channels missing from the lead field: {extra}; the noise sensors must be a subset of the lead field's sensors")
        return cls(lead_field, noise_cov.astype(np.float64), noise_names)

    def _forward_for(self, sensor: Sensor) -> ForwardModel:
        """Whitened forward-model state for exactly ``sensor``'s channels.

        Returns :attr:`forward` when it covers these channels in this order.

        Raises
        ------
        ValueError
            If a channel is missing from the lead field or the noise
            covariance; fitting needs both for every data channel.
        """
        names = list(sensor.names)
        if self.forward is not None and names == list(self.forward.sensor.names):
            return self.forward
        if self.lead_field is None:
            raise ValueError(f"data channels {names} do not match the forward model, and the estimator has no lead field to derive a matching one from; construct it with from_lead_field()")
        lead_field_names = set(self.lead_field.get_dim('sensor').names)
        missing = [name for name in names if name not in lead_field_names]
        if missing:
            raise ValueError(f"data channels missing from the lead field: {missing}; fitting requires a forward solution for every data channel")
        noise_channels = set(self.noise_channels)
        missing = [name for name in names if name not in noise_channels]
        if missing:
            raise ValueError(f"data channels missing from the noise covariance: {missing}; whitening requires a noise estimate for every fitted channel (were they excluded from the noise estimate as bad channels?)")
        lead_field = self.lead_field.sub(sensor=names)
        noise_channels = list(self.noise_channels)
        index = np.array([noise_channels.index(name) for name in names])
        noise_cov = self.noise_covariance[np.ix_(index, index)]
        if lead_field.has_dim('space'):
            g = lead_field.get_data(dims=('sensor', 'source', 'space')).astype(np.float64)
            g = g.reshape(g.shape[0], -1)
            space = lead_field.get_dim('space')
        else:
            g = lead_field.get_data(dims=('sensor', 'source')).astype(np.float64)
            space = None
        return ForwardModel(g, noise_cov, lead_field.get_dim('source'), lead_field.get_dim('sensor'), space)

    def __repr__(self) -> str:
        if self.forward is not None:
            return f'<{type(self).__name__}: {_forward_summary(self.forward)}>'
        if self.lead_field is None:
            return f'<{type(self).__name__}>'
        orientation = 'free' if self.lead_field.has_dim('space') else 'fixed'
        return f"<{type(self).__name__}: {_count_repr(len(self.lead_field.get_dim('source')), 'source')}, {_count_repr(len(self.lead_field.get_dim('sensor')), 'sensor')}, {orientation} orientation, {_count_repr(len(self.noise_channels), 'noise channel')}>"

    def fit_model(
            self,
            data: RegressionData,
            solver: Solver,
            verbose: bool = False,
    ) -> tuple[NCRF, SolverFit]:
        """Fit one fixed solver configuration on prepared, whitened data.

        The single-fit primitive underneath :meth:`fit`: no candidate selection,
        no scoring. Cross-validation uses it to fit the individual folds.

        Parameters
        ----------
        data
            Prepared, whitened data.
        solver
            Fixed solver configuration.
        verbose
            Print intermediate values of the cost functions.

        Raises
        ------
        ValueError
            If ``data``'s sensors do not match the forward model's, or if
            ``data`` is not whitened. The solver multiplies the lead field and
            the data by channel position and assumes isotropic noise, so either
            mismatch would silently produce wrong coefficients.
        """
        if self.forward is None:
            raise ValueError("estimator has no forward model; use fit(), which derives one from the data's channels")
        _assert_sensors_equal(data.sensor_dim.names, self.forward.sensor.names, 'data', 'forward model')
        if not data.is_whitened:
            raise ValueError("data is not whitened; use NCRFEstimator.fit(), which whitens the data, or whiten it with self.forward.whiten(data)")
        solver_fit = solver.solve(self.forward, data, verbose=verbose)
        model = NCRF(
            forward=self.forward,
            theta=solver_fit.theta,
            design=data.design,
        )
        return model, solver_fit

    def _score_fit(
            self,
            model: NCRF,
            solver_fit: SolverFit,
            data: RegressionData,
    ) -> dict[str, float]:
        """Scores for one fit on ``data``: the model metrics plus the solver's own.

        The same composition scores the training fit and every cross-validation
        fold, so candidate selection compares the very quantities the fit reports.
        """
        return merge_scores(model.evaluate(data), solver_fit.score(self.forward, data))

    def fit(
            self,
            data: RegressionData,
            solver: Solver,
            *,
            cv: CrossValidation | None = None,
            verbose: bool = False,
            compute_explained_variance: bool = False,
    ) -> NCRFFit:
        """Fit a configured solver to prepared regression data.

        Parameters
        ----------
        data
            Prepared M/EEG data and corresponding basis-projected covariates. The
            forward model is derived from the estimator's lead field and noise
            covariance for exactly the data's channels; a data channel missing
            from either is an error.
        solver
            Solver configuration. A solver with more than one configuration to
            choose from selects one through cross-validation before the final fit.
        cv
            Cross-validation configuration. Defaults to :class:`CrossValidation`;
            unused when the solver does not cross-validate.
        verbose
            If set True prints intermediate values of the cost functions (default ``False``).
        compute_explained_variance
            Compute the source-wise explained-variance diagnostic and store it
            on the result.

        Returns
        -------
        NCRFFit
            Fitted model, selected solver, solver state, training scores, and
            optional cross-validation and source-wise diagnostics.
        """
        forward = self._forward_for(data.sensor_dim)
        if forward is not self.forward:
            self = replace(self, forward=forward)
        data = self.forward.whiten(data)
        if cv is None:
            cv = CrossValidation()
        solver, cv_results = solver.search(self, data, cv)
        model, solver_fit = self.fit_model(data, solver, verbose)
        scores = self._score_fit(model, solver_fit, data)
        if compute_explained_variance:
            voxelwise = model.voxelwise_explained_variance(data)
        else:
            voxelwise = None

        return NCRFFit(
            model,
            solver=solver,
            solver_fit=solver_fit,
            scores=scores,
            voxelwise_explained_variance=voxelwise,
            cv_results=cv_results or None,
        )


class NCRFFit:
    """Report produced by :meth:`NCRFEstimator.fit`.

    Bundles the fitted :class:`NCRF` with the training-set evaluation and the
    fitting provenance. Model-level predictive quantities live on :attr:`model`;
    solver-specific state lives on :attr:`solver_fit`.

    Attributes
    ----------
    model
        The fitted :class:`NCRF` with TRF reconstruction, prediction, and
        evaluation methods.
    solver
        Immutable solver configuration used for the fit.
    solver_fit
        Solver-specific fitted state and iteration history (for ChampLasso,
        ``solver_fit.history``).
    scores
        Prediction metrics on the training data, keyed by name: the
        solver-independent model metrics plus whatever the solver contributes
        through :meth:`SolverFit.score` (for ChampLasso, ``cross_fit`` and
        ``weighted_l2_error``). For an arbitrary dataset use
        :meth:`NCRF.evaluate`.
    voxelwise_explained_variance
        Source-wise contributions to explained variance on the training data
        (``None`` unless requested at fit time).

    Notes
    -----
    Cross-validation scores are retained when the solver's search cross-validates,
    and are summarized by :meth:`cv_info`. The configuration the search settled on
    is :attr:`solver` itself (for ChampLasso, ``solver.mu``).
    """

    def __init__(
            self,
            model: NCRF,
            *,
            solver: Solver,
            solver_fit: SolverFit,
            scores: dict[str, float],
            voxelwise_explained_variance: NDVar | None,
            cv_results: list[CVResult] | None,
    ) -> None:
        self.model = model
        self.solver = solver
        self.solver_fit = solver_fit
        self.scores = scores
        self.voxelwise_explained_variance = voxelwise_explained_variance
        self._cv_results = cv_results

    def __repr__(self) -> str:
        forward = self.model.forward
        solver = type(self.solver).__name__
        scores = self.scores
        voxelwise = self.voxelwise_explained_variance is not None
        return f'<{type(self).__name__}: {_forward_summary(forward)}, solver={solver}, {scores=}, {voxelwise=}>'

    def cv_info(self) -> fmtxt.Table:
        """Summarize stored cross-validation scores in a table."""
        if not self._cv_results:
            raise ValueError("No cross-validation results; use a solver that searches over several configurations, such as ChampLasso(mu='auto').")
        return self.solver.cv_table(self._cv_results)
