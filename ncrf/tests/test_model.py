"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""
# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
from dataclasses import dataclass, replace
import logging
from importlib.metadata import version
import pickle
from unittest.mock import MagicMock, Mock

import mne
import numpy as np
import pytest

from ncrf import ChampLassoFit, CrossValidation, CVResult, ForwardModel, NCRF, NCRFEstimator, NCRFExecutionContext, RegressionData, Solver, SolverFit, TRFDesign, fit_model
from ncrf._linalg import gaussian_basis
from ncrf._solvers import champ_lasso
from ncrf._solvers.champ_lasso import ChampLassoHistory
from ncrf._metrics import merge_scores
from ncrf._model import _score_fit
from ncrf._typing import FloatArray
from eelbrain.trf import DeconvolutionData

import eelbrain
from eelbrain import Categorial, Datalist, NDVar, Scalar, Sensor, UTS, split_data


SENSOR = Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'c'])


def test_fit(monkeypatch):
    """fit_model() and NCRFEstimator.fit() work with any Solver, and a fixed configuration needs no cross-validation."""
    forward = Mock()
    solver = Mock()
    solver_fit = SolverFit(np.empty((2, 3)))
    solver.solve.return_value = solver_fit
    data = Mock(design=object(), forward=forward)
    model, returned_fit = fit_model(data, solver, True)
    assert model.forward is forward
    assert model.theta is solver_fit.theta
    assert model.design is data.design
    assert returned_fit is solver_fit
    solver.solve.assert_called_once_with(forward, data, verbose=True)

    # the solver assumes isotropic noise, so raw data must not reach it
    data.forward = None
    with pytest.raises(ValueError, match="data is not whitened"):
        fit_model(data, solver)

    # NCRFEstimator.fit() with a generic solver
    data = MagicMock()
    data.whiten.return_value = data
    forward = Mock(
        whitened_lead_field=np.ones((1, 1)),
        lead_field=np.ones((1, 1)),
        lead_field_scaling=1.0,
        sensor=SENSOR,
    )
    estimator = NCRFEstimator()
    monkeypatch.setattr(NCRFEstimator, '_forward_for', lambda self, sensor: forward)
    data.forward = forward
    data.sensor_dim = SENSOR
    data.design = Mock()
    data.responses = [np.arange(4, dtype=float)[None, :]]
    data.covariates = [np.ones((4, 1))]
    data.__iter__.side_effect = lambda: iter(zip(data.responses, data.covariates))
    data.__len__.return_value = 1
    solver = _ZeroSolver()
    result = estimator.fit(data, solver)
    assert result.solver is solver
    assert result.solver_fit.theta.shape == (1, 1)
    assert result.scores == {
        'explained_variance': pytest.approx(0),
        'l2_error': pytest.approx(1.75),  # 0.5 * (0 + 1 + 4 + 9) / 4 samples
    }
    prediction = result.model.predict(data)
    np.testing.assert_array_equal(prediction[0], np.zeros((1, 4)))
    # the fit records its execution context; no cross-validation, so no workers
    context = result.execution_context
    assert isinstance(context, NCRFExecutionContext)
    assert context.ncrf_version == version('ncrf')
    assert context.eelbrain_version == eelbrain.__version__
    assert context.duration >= 0
    assert context.n_workers is None

    # a searching solver decides the configuration and hands back its cv results
    searching_solver = Mock(spec=Solver)
    selected_solver = _ZeroSolver()
    cv_results = [Mock()]
    searching_solver.search.return_value = (selected_solver, cv_results)
    cv = CrossValidation(splits=4, n_workers=0)
    result = estimator.fit(data, searching_solver, cv=cv)
    assert result.solver is selected_solver
    assert result._cv_results is cv_results
    assert result.execution_context.n_workers == 0
    # search() gets what it needs to cross-validate on the configured folds
    searching_solver.search.assert_called_once_with(data, cv)

    # an Eelbrain Splits object stands in for a CrossValidation with those folds
    splits = split_data(np.array([[0, 100]]), partitions=4, test=0)
    searching_solver.search.reset_mock()
    estimator.fit(data, searching_solver, cv=splits)
    searching_solver.search.assert_called_once_with(data, CrossValidation(splits))

    # a solver implementing only solve() is a fixed configuration that needs no CV
    assert solver.without_history() is solver
    # no estimator or cv config is touched, since nothing is cross-validated
    assert solver.search(None, None) == (solver, [])

    # a generic table renders from whatever score keys are present
    cv_results = [
        CVResult(_ZeroSolver(), {'l2_error': 3.0, 'explained_variance': 0.1}),
        CVResult(solver, {'l2_error': 1.0, 'explained_variance': 0.4}),
    ]
    assert 'l2_error' in str(solver.cv_table(cv_results))


@dataclass(frozen=True)
class _ZeroSolver(Solver):
    def solve(self, forward, data, *, verbose=False):
        return SolverFit(np.zeros((1, 1)))


@dataclass(frozen=True)
class _ShapedZeroSolver(Solver):
    """Zero coefficients matching the forward model and design it is fit to."""

    def solve(self, forward, data, *, verbose=False):
        return SolverFit(np.zeros((forward.lead_field.shape[1], data.design.n_coefficients)))


def test_scores():
    """Training and fold scores combine the model metrics with the solver's own scores, which must not shadow them."""
    # training and cross-validation fold scores use the same composition
    model = Mock()
    model.evaluate.return_value = {'l2_error': 3.0, 'explained_variance': 0.5}
    solver_fit = Mock()
    solver_fit.score.return_value = {'cross_fit': 1.0}
    data = Mock(forward=object())
    scores = _score_fit(model, solver_fit, data)
    model.evaluate.assert_called_once_with(data)
    solver_fit.score.assert_called_once_with(data.forward, data)
    assert scores == {'l2_error': 3.0, 'explained_variance': 0.5, 'cross_fit': 1.0}

    # solvers without their own scores contribute nothing to the score dict
    assert SolverFit(np.empty((2, 3))).score(Mock(), Mock()) == {}

    # candidates are selected by score name, so a silent overwrite is not acceptable
    metrics = {'explained_variance': 0.5, 'l2_error': 3.0}
    assert merge_scores(metrics, {'cross_fit': 1.0}) == {**metrics, 'cross_fit': 1.0}
    with pytest.raises(ValueError, match="duplicate score l2_error"):
        merge_scores(metrics, {'cross_fit': 1.0, 'l2_error': 0.0})


def test_whiten():
    """whiten() attaches a forward model once, for the data's sensors, and only changes the responses."""
    raw = _synthetic_data()
    ete = raw.EtE
    forward = _forward(noise_covariance=np.eye(3) / 4)
    data = raw.whiten(forward)
    assert data.forward is forward
    # only the responses change, so arrays derived from the covariates carry over
    assert data._segment_covariates is raw._segment_covariates
    assert data.EtE is ete
    assert '_segment_responses' not in data.__dict__
    np.testing.assert_allclose(data.responses[0], 2 * raw.responses[0])
    np.testing.assert_array_equal(data.covariates[0], raw.covariates[0])

    # an equivalent forward model is a no-op, a different one an error
    assert data.whiten(forward) is data
    assert data.whiten(_forward(noise_covariance=np.eye(3) / 4)) is data
    with pytest.raises(ValueError, match="whitened with a different forward model"):
        data.whiten(_forward())
    # the same whitening filter with a different lead field is a different model:
    # whitened data is fit with the forward model it carries
    with pytest.raises(ValueError, match="whitened with a different forward model"):
        data.whiten(_forward(seed=2, noise_covariance=np.eye(3) / 4))
    with pytest.raises(ValueError, match="whitened with a different forward model"):
        data.whiten(ForwardModel(forward.lead_field[:, :2], np.eye(3) / 4, Scalar('source', range(2)), SENSOR, None))
    # the comparison is relative: at MEG scale, every value is below any absolute tolerance
    source = Scalar('source', range(4))
    meg_scale = raw.whiten(ForwardModel(forward.lead_field * 1e-8, np.eye(3) * 1e-26, source, SENSOR, None))
    assert meg_scale.whiten(ForwardModel(forward.lead_field * 1e-8, np.eye(3) * 1e-26, source, SENSOR, None)) is meg_scale
    with pytest.raises(ValueError, match="whitened with a different forward model"):
        meg_scale.whiten(ForwardModel(forward.lead_field * 1e-8, np.eye(3) * 5e-26, source, SENSOR, None))
    with pytest.raises(ValueError, match="whitened with a different forward model"):
        meg_scale.whiten(ForwardModel(forward.lead_field * 2e-8, np.eye(3) * 1e-26, source, SENSOR, None))

    # a forward model for other sensors cannot be attached, even when its whitening
    # filter is the same (as for a diagonal noise covariance with equal variances)
    other_sensor = Sensor([[0., 0, 1], [0, 1, 0], [1, 0, 0]], ['c', 'b', 'a'])
    other_forward = ForwardModel(np.ones((3, 4)), np.eye(3) / 4, Scalar('source', range(4)), other_sensor, None)
    with pytest.raises(ValueError, match="same channels in a different order"):
        raw.whiten(other_forward)
    with pytest.raises(ValueError, match="same channels in a different order"):
        data.whiten(other_forward)
    with pytest.raises(ValueError, match="same channels in a different order"):
        replace(raw, forward=other_forward)
    fewer_sensors = ForwardModel(np.ones((2, 4)), np.eye(2) / 4, Scalar('source', range(4)), SENSOR[:2], None)
    with pytest.raises(ValueError, match=r"only in data: \['c'\]"):
        data.whiten(fewer_sensors)


def _synthetic_data(
        scale: str | None = None,
        seed: int = 0,
        tstop: float = 0.05,
        names: tuple[str, str] = ('loud', 'quiet'),
        sensor: Sensor = SENSOR,
        n_segments: int = 1,
) -> RegressionData:
    """Two-predictor dataset on strongly mismatched stimulus scales."""
    rng = np.random.RandomState(seed)
    time = UTS(0, 0.01, 200)
    meg = NDVar(rng.normal(size=(n_segments, 3, 200)), ('case', sensor, time))
    stim = [
        NDVar(rng.normal(size=(n_segments, 200)) * 100 + 20, ('case', time), name=names[0]),
        NDVar(rng.normal(size=(n_segments, 200)) * 0.01 + 0.5, ('case', time), name=names[1]),
    ]
    if n_segments == 1:
        meg = meg[0]
        stim = [x[0] for x in stim]
    return RegressionData.from_data(meg, stim, 0, tstop, scale=scale)


def test_subset():
    """A subset's arrays are views into the full dataset's, its Gram matrices are normalized by its own sample count, and pickles carry only the inputs."""
    # a subset's arrays are views into the full dataset's, and its Gram matrices are
    # normalized by its own sample count
    data = _synthetic_data('l2', n_segments=2).whiten(_forward())
    assert data.blocks == [(0, 4, 200), (1, 4, 200)]  # lags 0-4 samples: the first 4 samples lack a full lag window
    assert len(data) == 2
    fold = data.subset([[10, 50], [195, 230]])
    assert fold.blocks == [(0, 10, 50), (0, 195, 200), (1, 4, 30)]
    assert fold.is_whitened
    assert fold.n_samples == 71
    assert fold.segment_n_samples == {0: 45, 1: 26}
    # the covariates are built once, on the source dataset, and sliced from there
    assert fold._segment_covariates is data._segment_covariates
    assert np.shares_memory(fold.covariates[0], data.covariates[0])
    np.testing.assert_array_equal(fold.covariates[0], data.covariates[0][6:46])
    np.testing.assert_array_equal(fold.covariates[2], data.covariates[1][:26])
    # the responses are views too, whitened once on the source dataset
    assert fold._segment_responses is data._segment_responses
    assert np.shares_memory(fold.responses[0], data.responses[0])
    np.testing.assert_array_equal(fold.responses[0], data.responses[0][:, 6:46])
    np.testing.assert_array_equal(fold.responses[2], data.responses[1][:, :26])
    # the Gram matrices apply the norm factor
    np.testing.assert_allclose(fold.bbt[0], fold.responses[0] @ fold.responses[0].T / fold.norm_factor ** 2)
    np.testing.assert_allclose(fold.bE[0], fold.responses[0] @ fold.covariates[0] / fold.norm_factor ** 2)
    np.testing.assert_allclose(fold.EtE[0], fold.covariates[0].T @ fold.covariates[0] / fold.norm_factor ** 2)
    # a subset of a subset
    assert fold.subset([[0, 30]]).blocks == [(0, 10, 30)]
    with pytest.raises(ValueError, match="contains none of the dataset's samples"):
        fold.subset([[100, 150]])

    # cutting segments into more blocks leaves the data term of the objective unchanged.
    # A cross-validation fold that holds out the middle of a segment trains on two
    # blocks of it, and has to be regularized like the complete data
    assert data.norm_factor ** 2 == data.n_samples
    # the same samples, each segment cut in two
    cut = data.subset([[4, 100], [100, 200], [204, 300], [300, 400]])
    assert cut.blocks == [(0, 4, 100), (0, 100, 200), (1, 4, 100), (1, 100, 200)]
    assert cut.n_samples == data.n_samples
    for key in ('bbt', 'bE', 'EtE'):
        np.testing.assert_allclose(sum(getattr(cut, key)), sum(getattr(data, key)), err_msg=key)

    # pickles carry the inputs and design; the solver's arrays are rebuilt on demand
    responses, covariates = data.responses, data.covariates  # populate the caches
    restored = pickle.loads(pickle.dumps(data))
    assert '_segment_responses' not in restored.__dict__ and '_segment_covariates' not in restored.__dict__
    assert restored.blocks == data.blocks
    np.testing.assert_array_equal(restored.forward.whitening_filter, data.forward.whitening_filter)
    np.testing.assert_allclose(restored.responses[0], responses[0])
    np.testing.assert_allclose(restored.covariates[0], covariates[0])


def test_from_data_validation(caplog):
    """Datasets have to match their design, flat or NaN M/EEG channels are rejected, and segments too short for the TRF are left out."""
    # directly constructed datasets have to match their design
    data = _synthetic_data()
    time = UTS(0, 0.01, 200)
    rng = np.random.RandomState(0)
    stim = NDVar(rng.normal(size=200), (time,), name='loud')
    with pytest.raises(ValueError, match="needs a sensor and a time dimension"):
        replace(data, data=DeconvolutionData(stim, NDVar(rng.normal(size=(3, 2, 200)), (SENSOR, Categorial('rep', ['a', 'b']), time))))
    with pytest.raises(ValueError, match="do not match the design"):
        replace(data, data=DeconvolutionData(stim, NDVar(rng.normal(size=(3, 200)), (SENSOR, time))))
    with pytest.raises(ValueError, match="no M/EEG data"):
        replace(data, data=DeconvolutionData(stim))
    with pytest.raises(ValueError, match="does not match design.tstep"):
        replace(data, design=replace(data.design, tstep=0.02))
    with pytest.raises(ValueError, match="segment index out of range"):
        replace(data, blocks=[(1, 0, 100)])
    with pytest.raises(ValueError, match="out of range for segment with 200 samples"):
        replace(data, blocks=[(0, 100, 201)])
    # the design only records a normalization that the predictors carry
    normalized = _synthetic_data('l2').design
    with pytest.raises(ValueError, match="records a centering that the predictors do not carry"):
        replace(data, design=normalized)
    with pytest.raises(ValueError, match="records a scaling that the predictors do not carry"):
        replace(data, design=replace(normalized, stim_baseline=None))
    # mismatched inputs are rejected by the Eelbrain data layer
    with pytest.raises(ValueError, match="same time dimension"):
        RegressionData.from_data(NDVar(rng.normal(size=(3, 200)), (SENSOR, time)), NDVar(stim.x, (UTS(1, 0.01, 200),)), 0, 0.05)

    # a flat channel (or NaN) corrupts the whitened data and the metrics, so it is
    # rejected up front
    x = rng.normal(size=(2, 3, 200))
    x[:, 1] = 0
    meg = Datalist([NDVar(x_seg, (SENSOR, time)) for x_seg in x])
    stim = Datalist([NDVar(rng.normal(size=200), (time,), name='x') for _ in meg])
    with pytest.raises(ValueError, match="flat data in <NDVar>-b"):
        RegressionData.from_data(meg, stim, 0, 0.05)
    x[:, 1] = rng.normal(size=(2, 200))
    x[1, 2, 10] = np.nan
    with pytest.raises(ValueError, match="NaN in <NDVar>-c"):
        RegressionData.from_data(meg, stim, 0, 0.05)

    # segments without a sample with a complete lag window are left out, with a warning
    # TRF from -0.1 to 0.5 s: lags -10 to 49, so segments need more than 59 samples
    times = [UTS(0, 0.01, n) for n in (300, 50, 59, 60)]
    meg = Datalist([NDVar(rng.normal(size=(3, len(time))), (SENSOR, time)) for time in times])
    stim = Datalist([NDVar(rng.normal(size=len(time)), (time,), name='x') for time in times])
    with caplog.at_level(logging.WARNING, 'ncrf._data'):
        data = RegressionData.from_data(meg, stim, -0.1, 0.5, scale=None)
    assert data.blocks == [(0, 49, 290), (3, 49, 50)]
    assert "2 segments not used" in caplog.text
    assert "longer than 59 samples: segment 1 (50 samples), 2 (59 samples)" in caplog.text
    # with padded predictors, all samples are kept
    caplog.clear()
    with caplog.at_level(logging.WARNING, 'ncrf._data'):
        data = RegressionData.from_data(meg, stim, -0.1, 0.5, scale=None, pad_stim=True)
    assert [i for i, _, _ in data.blocks] == [0, 1, 2, 3]
    assert not caplog.records


def _forward(seed: int = 1, noise_covariance: FloatArray | None = None) -> ForwardModel:
    """Forward model for the sensors of :func:`_synthetic_data`, with 4 sources."""
    rng = np.random.RandomState(seed)
    if noise_covariance is None:
        noise_covariance = np.eye(3)
    return ForwardModel(rng.normal(size=(3, 4)), noise_covariance, Scalar('source', range(4)), SENSOR, None)


def _model(design: TRFDesign, n_coefficients: int, seed: int = 1) -> NCRF:
    rng = np.random.RandomState(seed)
    return NCRF(_forward(seed), rng.normal(size=(4, n_coefficients)), design)


def test_champ_lasso_segments():
    """ChampLasso estimates and scores the sensor covariance per segment, from the segment's own samples."""
    # ChampLassoFit scores held-out data with the covariance of its own segment, not by
    # block position
    data = _synthetic_data('l2', n_segments=3).whiten(_forward())
    forward = data.forward
    theta = np.random.RandomState(2).normal(size=(4, data.design.n_coefficients))
    sigma_b = {i: np.eye(3) * (i + 1) for i in range(2)}  # fit on segments 0 and 1
    fit = ChampLassoFit(theta, ChampLassoHistory(), {}, sigma_b, {0: 196, 1: 196})

    # a block of the second segment (samples 200-400 on the concatenated axis)
    held_out = data.subset([[300, 350]])
    assert held_out.blocks == [(1, 100, 150)]
    scores = fit.score(forward, held_out)
    meg, covariates = held_out.responses[0], held_out.covariates[0]
    residual = (meg - forward.whitened_lead_field @ theta @ covariates.T) / held_out.norm_factor
    weighted_l2_error = 0.5 * (residual ** 2).sum() / 2  # sigma_b[1] = 2 I
    np.testing.assert_allclose(scores['weighted_l2_error'], weighted_l2_error)
    np.testing.assert_allclose(scores['cross_fit'], weighted_l2_error + 0.5 * np.log(np.linalg.det(sigma_b[1])))
    # the scores are per sample: 100 samples of segment 0 and 50 of segment 1
    held_out = data.subset([[100, 200], [300, 350]])
    assert held_out.blocks == [(0, 100, 200), (1, 100, 150)]
    scores = fit.score(forward, held_out)
    residuals = [meg - forward.whitened_lead_field @ theta @ covariates.T for meg, covariates in held_out]
    weighted_l2_error = (0.5 * (residuals[0] ** 2).sum() + 0.5 * (residuals[1] ** 2).sum() / 2) / 150
    np.testing.assert_allclose(scores['weighted_l2_error'], weighted_l2_error)
    logdet = (100 * np.log(np.linalg.det(sigma_b[0])) + 50 * np.log(np.linalg.det(sigma_b[1]))) / 150
    np.testing.assert_allclose(scores['cross_fit'], weighted_l2_error + 0.5 * logdet)

    # a segment the fit has no covariance for is scored with the training covariances
    # averaged over the training samples
    sigma_b = {0: np.eye(3), 1: np.diag([2., 3., 4.])}
    fit = ChampLassoFit(theta, ChampLassoHistory(), {}, sigma_b, n_samples={0: 150, 1: 50})
    pooled = (150 * sigma_b[0] + 50 * sigma_b[1]) / 200
    # computed when first needed
    assert 'sigma_b_pooled' not in fit.__dict__
    fit.score(forward, data.subset([[300, 350]]))
    assert 'sigma_b_pooled' not in fit.__dict__
    held_out = data.subset([[500, 550]])
    assert held_out.blocks == [(2, 100, 150)]
    scores = fit.score(forward, held_out)
    np.testing.assert_allclose(fit.sigma_b_pooled, pooled)
    meg, covariates = held_out.responses[0], held_out.covariates[0]
    residual = meg - forward.whitened_lead_field @ theta @ covariates.T
    weighted_l2_error = 0.5 * np.trace(np.linalg.solve(pooled, residual @ residual.T)) / 50
    np.testing.assert_allclose(scores['weighted_l2_error'], weighted_l2_error)
    np.testing.assert_allclose(scores['cross_fit'], weighted_l2_error + 0.5 * np.log(np.linalg.det(pooled)))
    # a pickled fit keeps its training sample counts
    assert pickle.loads(pickle.dumps(fit)).n_samples == {0: 150, 1: 50}

    # cross-validation that holds out whole segments (split='case')
    solver = champ_lasso.ChampLasso(mu=0.1, n_iter=1, n_iterc=1, n_iterf=2)
    # the fit records its training samples per segment
    fit = solver.solve(data.forward, data.subset([[0, 200], [300, 400]]))
    assert fit.n_samples == {0: 196, 1: 100}
    splits = split_data(data.data.segments, partitions=3, split='case', test=0)
    results = champ_lasso.crossvalidate(data, [solver, replace(solver, mu=0.2)], CrossValidation(splits, n_workers=0))
    assert len(results) == 2
    for result in results:
        assert np.isfinite(result.scores['cross_fit'])
        assert np.isfinite(result.scores['weighted_l2_error'])

    # champagne's per-segment seeds and residual covariances average over the segment's
    # own samples, whatever the other segments' lengths
    rng = np.random.RandomState(0)
    times = [UTS(0, 0.01, 300), UTS(0, 0.01, 100)]
    meg = Datalist([NDVar(rng.normal(size=(3, len(time))), (SENSOR, time)) for time in times])
    stim = Datalist([NDVar(rng.normal(size=len(time)), (time,), name='x') for time in times])
    data = RegressionData.from_data(meg, stim, 0, 0.05, scale=None).whiten(_forward())
    forward = data.forward
    theta = np.zeros((4, data.design.n_coefficients))
    gamma, sigma_b = champ_lasso._mne_seeds(forward, data)
    for i_segment in range(2):
        alone = RegressionData.from_data(meg[i_segment], stim[i_segment], 0, 0.05, scale=None).whiten(forward)
        assert alone.blocks == [(0, *data.blocks[i_segment][1:])]
        gamma_alone, sigma_b_alone = champ_lasso._mne_seeds(forward, alone)
        np.testing.assert_allclose(gamma[i_segment], gamma_alone[0])
        np.testing.assert_allclose(sigma_b[i_segment], sigma_b_alone[0])
        # the residual factor represents the segment's sample covariance
        yhat = champ_lasso._residual_factor(forward, theta, data, data.segment_blocks[i_segment])
        y = data.responses[i_segment]
        np.testing.assert_allclose(yhat @ yhat.T, y @ y.T / y.shape[1])


def _rank_deficient_covariance(rng: np.random.RandomState) -> tuple[FloatArray, FloatArray]:
    """Covariance of noise with one direction removed, as ICA component removal does; returns (covariance, projector)."""
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    projector = np.eye(3) - np.outer(u, u)
    noise = np.dot(projector, rng.normal(size=(3, 500)))
    return np.dot(noise, noise.T) / 500, projector


def test_forward_model():
    """ForwardModel whitens into the noise subspace, dropping negligible directions and rejecting channels without noise."""
    # a rank-deficient noise covariance whitens into its noise subspace rather than
    # being rejected
    rng = np.random.RandomState(0)
    covariance, projector = _rank_deficient_covariance(rng)
    forward = _forward(noise_covariance=covariance)

    assert forward.rank == 2
    assert forward.whitening_filter.shape == (2, 3)
    assert forward.whitened_lead_field.shape == (2, 4)
    np.testing.assert_allclose(np.dot(np.dot(forward.whitening_filter, covariance), forward.whitening_filter.T), np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(forward.whitened_noise_covariance, np.eye(2))
    np.testing.assert_allclose(np.linalg.norm(forward.whitened_lead_field, 2), 1)
    # the removed direction is invisible to the whitened space
    x = rng.normal(size=(3, 10))
    np.testing.assert_allclose(np.dot(forward.whitening_filter, x), np.dot(forward.whitening_filter, np.dot(projector, x)), atol=1e-12)
    assert '3 sensors (rank 2)' in repr(forward)
    assert '(rank' not in repr(_forward())

    # the rank in effect is pickled, and the derived quantities are reproduced
    restored = pickle.loads(pickle.dumps(forward))
    assert restored.rank == 2
    np.testing.assert_array_equal(restored.whitening_filter, forward.whitening_filter)

    # whitened data has rank channels; the rank is part of the forward model's identity
    data = _synthetic_data('l2').whiten(forward)
    assert data.responses[0].shape[0] == 2
    assert data.whiten(_forward(noise_covariance=covariance)) is data
    with pytest.raises(ValueError, match="already whitened with a different forward model"):
        data.whiten(_forward(noise_covariance=covariance + np.eye(3)))

    # directions far below the largest eigenvalue are dropped; measured ones are kept
    lead_field = np.ones((3, 4))
    source = Scalar('source', range(4))
    # eigenvectors spread over the channels, so that dropping one does not drop a channel
    q, _ = np.linalg.qr(np.random.RandomState(0).normal(size=(3, 3)))
    assert ForwardModel(lead_field, (q * [1., 1e-8, 1e-11]) @ q.T, source, SENSOR, None).rank == 3
    assert ForwardModel(lead_field, (q * [1., 1e-8, 1e-13]) @ q.T, source, SENSOR, None).rank == 2
    with pytest.raises(ValueError, match="no positive eigenvalues"):
        ForwardModel(lead_field, -np.eye(3), source, SENSOR, None)
    # a channel without noise is rejected rather than silently projected out, whether
    # it is exactly flat or merely far below the others
    with pytest.raises(ValueError, match="noise covariance has channels without noise: b"):
        ForwardModel(lead_field, np.diag([1., 0., 1.]), source, SENSOR, None)
    with pytest.raises(ValueError, match="noise covariance has channels without noise: b"):
        ForwardModel(lead_field, np.diag([1., 1e-20, 1.]), source, SENSOR, None)


def test_normalize():
    """normalize() applies a design's normalization or derives a scale, in place, as from_data() does."""
    # applying normalization to the covariates == applying it to the stimulus; the
    # scale derived from the dataset itself is what from_data uses
    for scale in ['l1', 'l2', 'spectral']:
        expected = _synthetic_data(scale)
        for design in (scale, expected.design):
            raw = _synthetic_data()
            stale = raw.covariates[0].copy()
            raw.EtE
            # the predictors are normalized in place, and the dataset itself is updated,
            # including derived arrays built before
            assert raw.normalize(design) is raw
            assert raw.design.scale == scale
            np.testing.assert_array_equal(raw.data.x.mean, expected.design.stim_baseline)
            np.testing.assert_array_equal(raw.design.stim_baseline, expected.design.stim_baseline)
            np.testing.assert_array_equal(raw.design.stim_scaling, expected.design.stim_scaling)
            assert not np.allclose(raw.covariates[0], stale)
            np.testing.assert_allclose(raw.covariates[0], expected.covariates[0])
            np.testing.assert_allclose(raw.EtE[0], expected.EtE[0])
    with pytest.raises(ValueError, match="need one of"):
        _synthetic_data().normalize('l3')
    with pytest.raises(ValueError, match="need None or one of"):
        _synthetic_data('l3')

    # the 'spectral' scale is the spectral norm of all samples' covariates per sample,
    # however they are divided into blocks
    data = _synthetic_data(n_segments=2)
    channels = np.split(np.vstack(data.covariates), np.cumsum(data.design.basis_widths)[:-1], axis=1)
    norms = data._spectral_norms()
    np.testing.assert_allclose(norms, [np.linalg.norm(E, 2) / np.sqrt(data.n_samples) for E in channels])
    # the same samples in blocks of unequal length
    pieces = data.subset([[0, 50], [50, 200], [200, 380], [380, 400]])
    np.testing.assert_allclose(pieces._spectral_norms(), norms)


def test_normalize_rejects():
    """Normalization that would corrupt the covariates or does not fit the dataset is rejected."""
    # the predictors are normalized in place, so a normalized dataset cannot be
    # normalized differently
    spectral = _synthetic_data('spectral')
    with pytest.raises(ValueError, match="already normalized, and can not be normalized with a different design"):
        spectral.normalize(_synthetic_data('l2').design)
    with pytest.raises(ValueError, match="already normalized with scale='spectral'"):
        spectral.normalize('l2')
    # applying the same normalization again is a no-op
    assert spectral.normalize(spectral.design) is spectral
    assert spectral.normalize(replace(spectral.design, basis_stride=None)) is spectral

    # a predictor without variation has no scale, and dividing by it destroys its
    # covariates; a single constant channel of a multi-channel predictor is enough,
    # and a NaN would silently poison every scale
    rng = np.random.RandomState(0)
    time = UTS(0, 0.01, 200)
    meg = NDVar(rng.normal(size=(3, 200)), (SENSOR, time))
    varying = NDVar(rng.normal(size=200), (time,), name='varying')
    constant = NDVar(np.full(200, 2.5), (time,), name='constant')
    x = rng.normal(size=(2, 200))
    x[1] = 7.
    invalid = [
        (constant, "x: flat data in constant"),
        (NDVar(np.full(200, 2.5), (time,)), "x: flat data in <NDVar>"),
        (NDVar(x, (Categorial('band', ['low', 'high']), time), name='bands'), "x: flat data in bands-high"),
        (NDVar(np.where(np.arange(200) == 7, np.nan, 1.), (time,), name='nan'), "x: NaN in nan"),
    ]
    for scale in ['l1', 'l2', 'spectral']:
        for stim, match in invalid:
            with pytest.raises(ValueError, match=match):
                RegressionData.from_data(meg, [varying, stim], 0, 0.05, scale=scale)
    # without scaling there is nothing to divide by
    data = RegressionData.from_data(meg, [varying, constant], 0, 0.05, scale=None)
    assert np.isfinite(data.covariates[0]).all()

    # a design for other predictors or lags
    data = _synthetic_data()
    with pytest.raises(ValueError, match="predictors"):
        data.normalize(_synthetic_data(names=('quiet', 'loud')).design)
    with pytest.raises(ValueError, match="tstop"):
        data.normalize(_synthetic_data(tstop=0.06).design)
    # scaling factors that would fill the covariates with NaN or infinity are rejected
    design = _synthetic_data('l2').design
    for factor in [0., np.nan, np.inf, -1.]:
        scaling = design.stim_scaling.copy()
        scaling[1] = factor
        with pytest.raises(ValueError, match=r"invalid 'l2' scaling \(quiet="):
            data.normalize(replace(design, stim_scaling=scaling))
    # the rejected designs left the data unchanged, so the valid scaling still applies
    np.testing.assert_allclose(data.normalize(design).covariates[0], _synthetic_data('l2').covariates[0])


def test_predict():
    """predict() and evaluate() require the fit's normalization and sensors, and weight blocks by their number of samples."""
    # coefficients fit on normalized covariates must not be applied to raw ones
    normalized = _synthetic_data('l2')
    model = _model(normalized.design, normalized.design.n_coefficients)
    raw = _synthetic_data()
    with pytest.raises(ValueError, match="different centering"):
        model.predict(raw)
    with pytest.raises(ValueError, match="different centering"):
        model.predict(_synthetic_data('l2', seed=2))
    with pytest.raises(ValueError, match="different scaling"):
        model.predict(raw.normalize(replace(normalized.design, stim_scaling=None, scale=None)))
    with pytest.raises(ValueError, match="predictors"):
        model.predict(_synthetic_data(names=('quiet', 'loud')))
    for expected, actual in zip(model.predict(normalized), model.predict(_synthetic_data().normalize(model.design))):
        np.testing.assert_allclose(expected, actual)

    # predict() is in the units of the MEG data; whitened=True gives the fitting space
    forward = model.forward
    predicted = model.predict(normalized)
    whitened = model.predict(normalized, whitened=True)
    for meg_scale, fit_scale in zip(predicted, whitened):
        # the whitened prediction is the prediction whitened like the responses
        np.testing.assert_allclose(np.dot(forward.whitening_filter, meg_scale), fit_scale)
    # whitening the input does not change the prediction, which only uses covariates
    for expected, actual in zip(predicted, model.predict(normalized.whiten(forward))):
        np.testing.assert_array_equal(expected, actual)

    # data whose sensors differ from the forward model must not be whitened silently
    # same channels in a different order: whitening would apply to the wrong channels
    reordered = _synthetic_data('l2', sensor=Sensor([[0., 0, 1], [0, 1, 0], [1, 0, 0]], ['c', 'b', 'a']))
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.predict(reordered)
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.evaluate(reordered)

    # a genuinely different channel set names the channels that differ
    renamed = _synthetic_data('l2', sensor=Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'z']))
    with pytest.raises(ValueError, match=r"only in data: \['z'\]; only in forward model: \['c'\]"):
        model.predict(renamed)
    with pytest.raises(ValueError, match="no lead field to derive a forward model from"):
        NCRFEstimator().fit(renamed, _ZeroSolver())

    # the metrics weight blocks by their number of samples, so a block's contribution
    # does not depend on how the data is cut
    data = _synthetic_data('l2', n_segments=2).whiten(_forward())
    model = _model(data.design, data.design.n_coefficients)
    a, b = data.subset([[10, 110]]), data.subset([[300, 350]])
    both = data.subset([[10, 110], [300, 350]])
    assert [stop - start for _, start, stop in both.blocks] == [100, 50]
    scores_a, scores_b, scores = model.evaluate(a), model.evaluate(b), model.evaluate(both)
    for key in ('explained_variance', 'l2_error'):
        np.testing.assert_allclose(scores[key], (100 * scores_a[key] + 50 * scores_b[key]) / 150)
    expected = (100 * model.voxelwise_explained_variance(a).x + 50 * model.voxelwise_explained_variance(b).x) / 150
    np.testing.assert_allclose(model.voxelwise_explained_variance(both).x, expected)
    # the whitened prediction is in the units of the whitened responses
    predicted = model.predict(both, whitened=True)
    np.testing.assert_allclose(predicted[1], model.forward.whitened_lead_field @ model.theta @ both.covariates[1].T)
    np.testing.assert_allclose(scores['l2_error'], sum(0.5 * ((meg - p) ** 2).sum() for (meg, _), p in zip(both, predicted)) / 150)


def test_estimator_forward():
    """NCRFEstimator accepts the noise in any form, and derives the forward model for the data's channels."""
    # every form of noise input yields the covariance of the lead field's sensors
    rng = np.random.RandomState(0)
    lead_field = NDVar(rng.normal(size=(3, 4)), (SENSOR, Scalar('source', range(4))))
    names = list(SENSOR.names)
    a = rng.normal(size=(3, 3))
    data = a.dot(a.T)
    covariance = mne.Covariance(data, names, [], [], 0)
    np.testing.assert_array_equal(NCRFEstimator.from_lead_field(lead_field, covariance).noise_covariance, data)

    # a diagonal covariance is expanded to a full matrix
    variance = np.diag(data).copy()
    diagonal = mne.Covariance(variance, names, [], [], 0)
    np.testing.assert_array_equal(NCRFEstimator.from_lead_field(lead_field, diagonal).noise_covariance, np.diag(variance))

    # empty-room data is reduced to its covariance
    empty_room = NDVar(rng.normal(size=(3, 500)), (SENSOR, UTS(0, 0.01, 500)))
    x = empty_room.get_data(('sensor', 'time'))
    np.testing.assert_allclose(NCRFEstimator.from_lead_field(lead_field, empty_room).noise_covariance, x.dot(x.T) / x.shape[1])

    # the covariance is stored as given; channel order is aligned when the forward model is derived
    reverse = np.ix_([2, 1, 0], [2, 1, 0])
    reversed_covariance = mne.Covariance(data[reverse], names[::-1], [], [], 0)
    estimator = NCRFEstimator.from_lead_field(lead_field, reversed_covariance)
    assert list(estimator.noise_channels) == names[::-1]
    np.testing.assert_array_equal(estimator.noise_covariance, data[reverse])
    np.testing.assert_array_equal(estimator._forward_for(SENSOR).noise_covariance, data)
    reordered_room = NDVar(x[::-1], (Sensor([[0., 0, 1], [0, 1, 0], [1, 0, 0]], names[::-1]), UTS(0, 0.01, 500)))
    np.testing.assert_allclose(NCRFEstimator.from_lead_field(lead_field, reordered_room)._forward_for(SENSOR).noise_covariance, x.dot(x.T) / x.shape[1])

    # noise for a subset of the lead field's channels is stored as is;
    # the full forward solution is retained
    estimator = NCRFEstimator.from_lead_field(lead_field, mne.Covariance(data[:2, :2], names[:2], [], [], 0))
    assert list(estimator.noise_channels) == names[:2]
    np.testing.assert_array_equal(estimator.noise_covariance, data[:2, :2])
    assert estimator.lead_field is lead_field

    # noise for channels beyond the lead field points to a data error; an extra channel is not silently dropped
    with pytest.raises(ValueError, match=r"noise covariance channels missing from the lead field: \['x'\]"):
        NCRFEstimator.from_lead_field(lead_field, mne.Covariance(np.eye(4), [*names, 'x'], [], [], 0))
    # a bare array has no channel names to align by
    with pytest.raises(TypeError, match="Invalid noise type"):
        NCRFEstimator.from_lead_field(lead_field, np.eye(3))
    with pytest.raises(TypeError, match="Invalid noise type"):
        NCRFEstimator.from_lead_field(lead_field, 'noise-cov.fif')

    # data on a subset of the forward model's channels is fit with a matching sub-
    # forward
    noise = covariance
    estimator = NCRFEstimator.from_lead_field(lead_field, noise)
    time = UTS(0, 0.01, 200)
    sensor_sub = Sensor([[0., 0, 1], [1., 0, 0]], ['c', 'a'])
    meg = NDVar(rng.normal(size=(2, 200)), (sensor_sub, time))
    stim = NDVar(rng.normal(size=200), (time,), name='x')
    data = RegressionData.from_data(meg, stim, 0, 0.05, scale=None)
    result = estimator.fit(data, _ShapedZeroSolver())
    # the model's forward is derived for the data's channels, in the data's order
    model = result.model
    assert list(model.forward.sensor.names) == ['c', 'a']
    np.testing.assert_array_equal(model.forward.lead_field, lead_field.x[[2, 0]])
    np.testing.assert_array_equal(model.forward.noise_covariance, noise.data[np.ix_([2, 0], [2, 0])])

    # the derived forward is reused for further fits to the same channels, but
    # does not travel with a pickled estimator
    assert estimator._forward_for(sensor_sub) is model.forward
    assert estimator.fit(data, _ShapedZeroSolver()).model.forward is model.forward
    assert '_forwards' not in pickle.loads(pickle.dumps(estimator)).__dict__

    # the derived forward is identical to one built from the trimmed inputs directly
    reference = NCRFEstimator.from_lead_field(lead_field.sub(sensor=['c', 'a']), mne.Covariance(noise.data[np.ix_([2, 0], [2, 0])], ['c', 'a'], [], [], 0))._forward_for(sensor_sub)
    np.testing.assert_array_equal(model.forward.whitening_filter, reference.whitening_filter)
    np.testing.assert_array_equal(model.forward.whitened_lead_field, reference.whitened_lead_field)

    # data with channels the lead field does not cover is an error naming the lead field
    sensor_extra = Sensor([[1., 0, 0], [0, 0.5, 0.5]], ['a', 'z'])
    meg_extra = NDVar(rng.normal(size=(2, 200)), (sensor_extra, time))
    data_extra = RegressionData.from_data(meg_extra, stim, 0, 0.05, scale=None)
    with pytest.raises(ValueError, match=r"data channels missing from the lead field: \['z'\]"):
        estimator.fit(data_extra, _ShapedZeroSolver())

    # data whitened with another model's forward is not silently fit with that
    # lead field instead of the estimator's
    other_forward = ForwardModel(rng.normal(size=(2, 4)), noise.data[np.ix_([2, 0], [2, 0])], Scalar('source', range(4)), sensor_sub, None)
    with pytest.raises(ValueError, match="whitened with a different forward model"):
        estimator.fit(data.whiten(other_forward), _ShapedZeroSolver())
    assert estimator.fit(data.whiten(model.forward), _ShapedZeroSolver()).model.forward is model.forward

    # a channel with a lead field but no noise estimate blames the noise covariance
    estimator_sub = NCRFEstimator.from_lead_field(lead_field, mne.Covariance(noise.data[:2, :2], names[:2], [], [], 0))
    meg_full = NDVar(rng.normal(size=(3, 200)), (SENSOR, time))
    data_full = RegressionData.from_data(meg_full, stim, 0, 0.05, scale=None)
    with pytest.raises(ValueError, match=r"data channels missing from the noise covariance: \['c'\]"):
        estimator_sub.fit(data_full, _ShapedZeroSolver())


def test_h():
    """h is the full TRF on the time axis of the fit lags, and h_scaled restores the stimulus scale."""
    # h_scaled restores the original stimulus scale, whichever scaling was used
    for scale in [None, 'l1', 'l2', 'spectral']:
        data = _synthetic_data(scale)
        model = _model(data.design, data.design.n_coefficients)
        h, h_scaled = model.h, model.h_scaled
        if scale is None:
            assert h_scaled is h
            continue
        for i, factor in enumerate(data.design.stim_scaling):
            np.testing.assert_allclose(h_scaled[i].x, h[i].x / factor)

    # h_scaled equals the h of an equivalent model whose covariates were not scaled
    scaled = _synthetic_data('l2')
    design = scaled.design
    centered = _synthetic_data().normalize(replace(design, stim_scaling=None, scale=None))
    model = _model(design, design.n_coefficients)
    # coefficients on the unscaled covariates that make the same predictions
    unscaled_model = NCRF(model.forward, model.theta / design.expand(design.stim_scaling), centered.design)
    for expected, actual in zip(model.predict(scaled), unscaled_model.predict(centered)):
        np.testing.assert_allclose(expected, actual)
    # h of the unscaled fit is already in stimulus units
    assert unscaled_model.h_scaled is unscaled_model.h
    for expected, actual in zip(unscaled_model.h, model.h_scaled):
        np.testing.assert_allclose(actual.x, expected.x)

    # a predictor with a single basis function still expands into a full TRF
    rng = np.random.RandomState(0)
    time = UTS(0, 0.01, 200)
    meg = NDVar(rng.normal(size=(3, 200)), (SENSOR, time))
    bands = Categorial('band', ['low', 'high'])
    onsets = Categorial('onset', ['on', 'off', 'both'])
    for tstop, n_atoms in [(0.03, 1), (0.06, 4)]:
        n_lags = int(round(tstop / 0.01))  # tstop is exclusive
        for stim, dimnames, shape in [
            (NDVar(rng.normal(size=200), (time,), name='x'), ('source', 'time'), (4, n_lags)),
            (NDVar(rng.normal(size=(2, 200)), (bands, time), name='x'), ('band', 'source', 'time'), (2, 4, n_lags)),
            (NDVar(rng.normal(size=(2, 3, 200)), (bands, onsets, time), name='x'), ('band', 'onset', 'source', 'time'), (2, 3, 4, n_lags)),
        ]:
            data = RegressionData.from_data(meg, stim, 0, tstop, scale=None)
            assert data.design.basis[0].shape[1] == n_atoms
            h = _model(data.design, data.design.n_coefficients).h
            assert h.dimnames == dimnames
            assert h.shape == shape
            np.testing.assert_allclose(h.time.times, np.arange(n_lags) * 0.01)
            # the scaling is undone along the predictor's feature dimensions
            scaled = RegressionData.from_data(meg, stim, 0, tstop, scale='l2')
            h_scaled = _model(scaled.design, scaled.design.n_coefficients).h_scaled
            assert h_scaled.dimnames == dimnames
            np.testing.assert_allclose(h_scaled.x, h.x / scaled.design.stim_scaling.reshape((*stim.shape[:-1], 1, 1)))

    # the TRF's time axis is that of the lags that were fit, i.e. tstart rounded to the
    # sample grid
    stim = NDVar(rng.normal(size=200), (time,), name='x')
    data = RegressionData.from_data(meg, stim, -0.104, 0.053, scale=None)
    assert data.design.start_samples == [-10]
    assert data.design.filter_length == [16]
    h = _model(data.design, data.design.n_coefficients).h
    np.testing.assert_allclose(h.time.times, np.arange(-10, 6) * 0.01)


def test_design():
    """TRF timing per predictor, and the Gaussian basis."""
    # a scalar TRF time applies to all predictors, a sequence has one value per
    # predictor
    time = UTS(0, 0.01, 200)
    stim = [NDVar(np.zeros(200), (time,), name='a'), NDVar(np.zeros(200), (time,), name='b')]
    design = TRFDesign.from_data(DeconvolutionData(stim), -0.1, [0.2, 0.3])
    assert design.tstart == [-0.1, -0.1]
    assert design.tstop == [0.2, 0.3]
    # a one-element sequence is not broadcast
    with pytest.raises(ValueError, match="need one value per predictor"):
        TRFDesign.from_data(DeconvolutionData(stim), [-0.1], 0.2)

    # the Gaussian basis does not depend on the time offset
    basis = gaussian_basis(4, np.linspace(0, 1, 11), 0.1)
    shifted_basis = gaussian_basis(4, np.linspace(10, 11, 11), 0.1)
    assert basis.shape == (11, 4)
    np.testing.assert_allclose(basis, shifted_basis)
