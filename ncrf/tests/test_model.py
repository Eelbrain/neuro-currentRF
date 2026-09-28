"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""
# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
from dataclasses import dataclass, replace
from unittest.mock import MagicMock, Mock

import mne
import numpy as np
import pytest

from ncrf import CrossValidation, CVResult, ForwardModel, NCRF, NCRFEstimator, RegressionData, Solver, SolverFit, TRFDesign
from ncrf._data import covariate_from_stim
from ncrf._linalg import gaussian_basis
from ncrf._metrics import merge_scores
from .fetch import load

from eelbrain import Categorial, NDVar, Scalar, Sensor, UTS, concatenate


SENSOR = Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'c'])


def test_fit_model():
    forward = Mock()
    forward.sensor.names = ['a', 'b']
    estimator = NCRFEstimator(forward=forward)
    solver = Mock()
    solver_fit = SolverFit(np.empty((2, 3)))
    solver.solve.return_value = solver_fit
    data = Mock(design=object(), is_whitened=True)
    data.sensor_dim.names = ['a', 'b']

    model, returned_fit = estimator.fit_model(data, solver, True)

    assert model.forward is estimator.forward
    assert model.theta is solver_fit.theta
    assert model.design is data.design
    assert returned_fit is solver_fit
    solver.solve.assert_called_once_with(estimator.forward, data, verbose=True)

    # channel misattribution would be silent, so mismatched sensors are rejected
    data.sensor_dim.names = ['b', 'a']
    with pytest.raises(ValueError, match="data sensors do not match the forward model"):
        estimator.fit_model(data, solver)
    data.sensor_dim.names = ['a', 'b']

    # the solver assumes isotropic noise, so raw data must not reach it
    data.is_whitened = False
    with pytest.raises(ValueError, match="data is not whitened"):
        estimator.fit_model(data, solver)


@dataclass(frozen=True)
class _ZeroSolver(Solver):
    def solve(self, forward, data, *, verbose=False):
        return SolverFit(np.zeros((1, 1)))


@dataclass(frozen=True)
class _ShapedZeroSolver(Solver):
    """Zero coefficients matching the forward model and design it is fit to."""

    def solve(self, forward, data, *, verbose=False):
        return SolverFit(np.zeros((forward.lead_field.shape[1], data.design.n_coefficients)))


def test_fit_accepts_generic_solver(monkeypatch):
    data = MagicMock()
    estimator = NCRFEstimator(forward=Mock(
        whiten=Mock(return_value=data),
        whitened_lead_field=np.ones((1, 1)),
        lead_field=np.ones((1, 1)),
        lead_field_scaling=1.0,
        sensor=SENSOR,
    ))
    data.sensor_dim = SENSOR
    data.design = Mock()
    data.covariates = [np.ones((4, 1))]
    data.norm_factor = 1.0
    data.__iter__.side_effect = lambda: iter([
        (np.arange(4, dtype=float)[None, :], np.ones((4, 1))),
    ])
    data.__len__.return_value = 1
    solver = _ZeroSolver()

    result = estimator.fit(data, solver)

    assert result.solver is solver
    assert result.solver_fit.theta.shape == (1, 1)
    assert result.scores == {
        'explained_variance': pytest.approx(0),
        'l2_error': pytest.approx(7),
    }
    prediction = result.model.predict(data)
    np.testing.assert_array_equal(prediction[0], np.zeros((1, 4)))

    # a searching solver decides the configuration and hands back its cv results
    searching_solver = Mock(spec=Solver)
    selected_solver = _ZeroSolver()
    cv_results = [Mock()]
    searching_solver.search.return_value = (selected_solver, cv_results)
    cv = CrossValidation(n_splits=4, n_workers=0)

    result = estimator.fit(data, searching_solver, cv=cv)

    assert result.solver is selected_solver
    assert result._cv_results is cv_results
    # search() gets what it needs to cross-validate on the configured folds
    searching_solver.search.assert_called_once_with(estimator, data, cv)


def test_default_search_contract():
    """A solver implementing only solve() is a fixed configuration that needs no CV."""
    solver = _ZeroSolver()

    assert solver.without_history() is solver
    # no estimator or cv config is touched, since nothing is cross-validated
    assert solver.search(None, None, None) == (solver, [])

    # a generic table renders from whatever score keys are present
    cv_results = [
        CVResult(_ZeroSolver(), {'l2_error': 3.0, 'explained_variance': 0.1}),
        CVResult(solver, {'l2_error': 1.0, 'explained_variance': 0.4}),
    ]
    assert 'l2_error' in str(solver.cv_table(cv_results))


def test_score_fit():
    """Training and cross-validation fold scores use the same composition."""
    estimator = NCRFEstimator(forward=object())
    model = Mock()
    model.evaluate.return_value = {'l2_error': 3.0, 'explained_variance': 0.5}
    solver_fit = Mock()
    solver_fit.score.return_value = {'cross_fit': 1.0}
    data = object()

    scores = estimator._score_fit(model, solver_fit, data)

    model.evaluate.assert_called_once_with(data)
    solver_fit.score.assert_called_once_with(estimator.forward, data)
    assert scores == {'l2_error': 3.0, 'explained_variance': 0.5, 'cross_fit': 1.0}


def test_solver_fit_score_defaults_empty():
    """Solvers without their own scores contribute nothing to the score dict."""
    assert SolverFit(np.empty((2, 3))).score(Mock(), Mock()) == {}


def test_merge_scores_rejects_shadowing():
    """Candidates are selected by score name, so a silent overwrite is not acceptable."""
    metrics = {'explained_variance': 0.5, 'l2_error': 3.0}

    assert merge_scores(metrics, {'cross_fit': 1.0}) == {**metrics, 'cross_fit': 1.0}
    with pytest.raises(ValueError, match="duplicate score l2_error"):
        merge_scores(metrics, {'cross_fit': 1.0, 'l2_error': 0.0})


def test_whitening_guard():
    data = RegressionData.__new__(RegressionData)
    whitening_filter = np.eye(3) * 2
    data.whitener = whitening_filter
    data.sensor_dim = SENSOR

    # the same filter is a no-op, a different filter an error
    assert data.whiten(whitening_filter) is data
    assert data.whiten(whitening_filter.copy()) is data
    with pytest.raises(ValueError, match="whitened with a different filter"):
        data.whiten(whitening_filter * 2)

    forward = _forward()
    with pytest.raises(ValueError, match="whitened with a different filter"):
        forward.whiten(data)
    data.whitener = forward.whitening_filter
    assert forward.whiten(data) is data


def _synthetic_data(
        scale: str | None = None,
        seed: int = 0,
        tstop: float = 0.05,
        names: tuple[str, str] = ('loud', 'quiet'),
) -> RegressionData:
    """Two-predictor dataset on strongly mismatched stimulus scales."""
    rng = np.random.RandomState(seed)
    time = UTS(0, 0.01, 200)
    meg = [NDVar(rng.normal(size=(3, 200)), (SENSOR, time))]
    stim = [[
        NDVar(rng.normal(size=200) * 100 + 20, (time,), name=names[0]),
        NDVar(rng.normal(size=200) * 0.01 + 0.5, (time,), name=names[1]),
    ]]
    return RegressionData.from_data(meg, stim, 0, tstop, scale=scale)


def test_timeslice_boolean_mask():
    data = _synthetic_data()
    mask = np.zeros(data.meg[0].shape[1], dtype=bool)
    mask[10:50] = True

    by_mask = data.timeslice(mask)
    by_index = data.timeslice(np.flatnonzero(mask))

    np.testing.assert_array_equal(by_mask.meg[0], by_index.meg[0])
    np.testing.assert_array_equal(by_mask.covariates[0], by_index.covariates[0])
    assert by_mask.norm_factor == by_index.norm_factor


def test_rejects_inconsistent_arrays():
    """Directly constructed datasets have to match their metadata."""
    data = _synthetic_data()
    meg, covariates = data.meg[0], data.covariates[0]

    with pytest.raises(ValueError, match="1 MEG segment but 2 covariate matrices"):
        replace(data, covariates=[covariates, covariates])
    with pytest.raises(ValueError, match="segment 0: MEG has 2 channels, sensor_dim 3 sensors"):
        replace(data, meg=[meg[:2]])
    with pytest.raises(ValueError, match="segment 0: covariates have shape"):
        replace(data, covariates=[covariates[1:]])
    # e.g., the lag matrix instead of its basis projection
    with pytest.raises(ValueError, match="were they projected onto the basis"):
        replace(data, covariates=[np.ones((len(covariates), data.design.n_coefficients + 2))])


def _forward(seed: int = 1) -> ForwardModel:
    """Forward model for the sensors of :func:`_synthetic_data`, with 4 sources."""
    rng = np.random.RandomState(seed)
    return ForwardModel(rng.normal(size=(3, 4)), np.eye(3), Scalar('source', range(4)), SENSOR, None)


def _model(design: TRFDesign, n_coefficients: int, seed: int = 1) -> NCRF:
    rng = np.random.RandomState(seed)
    return NCRF(_forward(seed), rng.normal(size=(4, n_coefficients)), design)


@pytest.mark.parametrize('scale', ['l1', 'l2', 'spectral'])
def test_normalize_matches_from_data(scale):
    """Applying normalization to covariates == applying it to the stimulus."""
    expected = _synthetic_data(scale)
    raw = _synthetic_data()
    assert not np.allclose(raw.covariates[0], expected.covariates[0])

    normalized = raw.normalize(expected.design)
    np.testing.assert_allclose(normalized.covariates[0], expected.covariates[0])
    # the source dataset is unchanged
    np.testing.assert_allclose(raw.covariates[0], _synthetic_data().covariates[0])


def test_normalize_does_not_write_through_shared_covariates():
    """Datasets sharing covariate arrays must not be normalized behind each other's back."""
    data = _synthetic_data()
    design = _synthetic_data('l2').design
    before = data.covariates[0].copy()

    whitened = data.whiten(np.eye(3))
    assert whitened.covariates[0] is data.covariates[0]  # whitening only copies meg
    normalized = whitened.normalize(design)

    np.testing.assert_allclose(normalized.covariates[0], _synthetic_data('l2').covariates[0])
    # the dataset the whitened view was derived from is untouched
    np.testing.assert_array_equal(data.covariates[0], before)
    assert data.design.stim_scaling is None


def test_normalize_rejects_renormalization():
    data = _synthetic_data('l2')
    other = _synthetic_data('l2', seed=2)

    with pytest.raises(ValueError, match="carry different centering"):
        data.normalize(other.design)
    # applying the same normalization again is a no-op
    np.testing.assert_array_equal(data.normalize(data.design).covariates[0], data.covariates[0])


@pytest.mark.parametrize('scale', ['l1', 'l2', 'spectral'])
def test_rejects_constant_predictor(scale):
    """A predictor without variation has no scale, and dividing by it destroys its covariates."""
    rng = np.random.RandomState(0)
    time = UTS(0, 0.01, 200)
    meg = [NDVar(rng.normal(size=(3, 200)), (SENSOR, time))]
    varying = NDVar(rng.normal(size=200), (time,), name='varying')
    constant = NDVar(np.full(200, 2.5), (time,), name='constant')
    unnamed = NDVar(np.full(200, 2.5), (time,))
    # a single constant channel of a multi-channel predictor is enough
    x = rng.normal(size=(2, 200))
    x[1] = 7.
    bands = NDVar(x, (Categorial('band', ['low', 'high']), time), name='bands')

    with pytest.raises(ValueError, match="constant: predictor is constant over time"):
        RegressionData.from_data(meg, [[varying, constant]], 0, 0.05, scale=scale)
    with pytest.raises(ValueError, match="<unnamed>: predictor is constant over time"):
        RegressionData.from_data(meg, [[varying, unnamed]], 0, 0.05, scale=scale)
    with pytest.raises(ValueError, match=r"bands\[1\]: predictor is constant over time"):
        RegressionData.from_data(meg, [[varying, bands]], 0, 0.05, scale=scale)

    # without scaling there is nothing to divide by
    data = RegressionData.from_data(meg, [[varying, constant]], 0, 0.05, scale=None)
    assert np.isfinite(data.covariates[0]).all()


def test_from_data_validates_before_modifying():
    """Invalid input is rejected before in_place=True rescales any segment."""
    rng = np.random.RandomState(0)
    time = UTS(0, 0.01, 200)
    x = rng.normal(size=(3, 200))
    flat = rng.normal(size=(3, 200))
    flat[1] = 0
    meg = [NDVar(x.copy(), (SENSOR, time)), NDVar(flat, (SENSOR, time))]
    stim = [[NDVar(rng.normal(size=200), (time,), name='x')] for _ in meg]
    constant = NDVar(np.full(200, 2.5), (time,), name='constant')

    with pytest.raises(ValueError, match=r"segment 1 has flat channels \(b\)"):
        RegressionData.from_data(meg, stim, 0, 0.05, in_place=True)
    np.testing.assert_array_equal(meg[0].x, x)
    with pytest.raises(ValueError, match="constant: predictor is constant over time"):
        RegressionData.from_data(meg[:1], [[constant]], 0, 0.05, in_place=True)
    np.testing.assert_array_equal(meg[0].x, x)


@pytest.mark.parametrize('factor', [0., np.nan, np.inf, -1.])
def test_normalize_rejects_invalid_scaling(factor):
    """Scaling factors that would fill the covariates with NaN or infinity are rejected."""
    data = _synthetic_data()
    design = _synthetic_data('l2').design
    scaling = design.stim_scaling.copy()
    scaling[1] = factor

    with pytest.raises(ValueError, match=r"invalid 'l2' scaling \(quiet="):
        data.normalize(replace(design, stim_scaling=scaling))
    # the valid scaling still applies
    np.testing.assert_allclose(data.normalize(design).covariates[0], _synthetic_data('l2').covariates[0])


def test_normalize_rejects_incompatible_design():
    data = _synthetic_data()

    with pytest.raises(ValueError, match="stim_names"):
        data.normalize(_synthetic_data(names=('quiet', 'loud')).design)
    with pytest.raises(ValueError, match="tstop"):
        data.normalize(_synthetic_data(tstop=0.06).design)


def test_predict_requires_fit_normalization():
    """Coefficients fit on normalized covariates must not be applied to raw ones."""
    normalized = _synthetic_data('l2')
    model = _model(normalized.design, normalized.design.n_coefficients)
    raw = _synthetic_data()

    with pytest.raises(ValueError, match="different centering"):
        model.predict(raw)
    with pytest.raises(ValueError, match="different scaling"):
        model.predict(raw.normalize(replace(normalized.design, stim_scaling=None, scale=None)))
    with pytest.raises(ValueError, match="stim_names"):
        model.predict(_synthetic_data(names=('quiet', 'loud')))

    for expected, actual in zip(model.predict(normalized), model.predict(raw.normalize(model.design))):
        np.testing.assert_allclose(expected, actual)


def test_predict_returns_meg_scale():
    """predict() is in the units of the MEG data; whitened=True gives the fitting space."""
    data = _synthetic_data('l2')
    model = _model(data.design, data.design.n_coefficients)
    forward = model.forward

    predicted = model.predict(data)
    whitened = model.predict(data, whitened=True)

    assert not np.allclose(predicted[0], whitened[0])
    for meg_scale, fit_scale in zip(predicted, whitened):
        # whitening and dividing by sqrt(n_times) is exactly what from_data() applied
        np.testing.assert_allclose(np.dot(forward.whitening_filter, meg_scale) / data.norm_factor, fit_scale)
    # whitening the input does not change the prediction, which only uses covariates
    for expected, actual in zip(predicted, model.predict(forward.whiten(data))):
        np.testing.assert_array_equal(expected, actual)


def test_predict_rejects_mismatched_normalization():
    design = _synthetic_data('l2').design
    model = _model(design, design.n_coefficients)
    other = _synthetic_data('l2', seed=2)

    with pytest.raises(ValueError, match="different centering"):
        model.predict(other)


def test_rejects_sensor_mismatch():
    """Data whose sensors differ from the forward model must not be whitened silently."""
    data = _synthetic_data('l2')
    # same channels in a different order: whitening would apply to the wrong channels
    reordered = replace(data, sensor_dim=Sensor([[0., 0, 1], [0, 1, 0], [1, 0, 0]], ['c', 'b', 'a']))

    model = _model(data.design, data.design.n_coefficients)
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.predict(reordered)
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.evaluate(reordered)

    # a genuinely different channel set names the channels that differ
    renamed = replace(data, sensor_dim=Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'z']))
    with pytest.raises(ValueError, match=r"only in data: \['z'\]; only in forward model: \['c'\]"):
        model.predict(renamed)

    estimator = NCRFEstimator(forward=model.forward)
    with pytest.raises(ValueError, match="no lead field to derive a matching one from"):
        estimator.fit(renamed, _ZeroSolver())


def test_estimator_noise_forms():
    """Every form of noise input yields the covariance of the lead field's sensors."""
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
    assert estimator.forward is None


def test_estimator_rejects_invalid_noise():
    """Noise for channels beyond the lead field points to a data error."""
    rng = np.random.RandomState(0)
    lead_field = NDVar(rng.normal(size=(3, 4)), (SENSOR, Scalar('source', range(4))))
    names = list(SENSOR.names)

    # an extra channel is not silently dropped
    with pytest.raises(ValueError, match=r"noise covariance channels missing from the lead field: \['x'\]"):
        NCRFEstimator.from_lead_field(lead_field, mne.Covariance(np.eye(4), [*names, 'x'], [], [], 0))
    # a bare array has no channel names to align by
    with pytest.raises(TypeError, match="Invalid noise type"):
        NCRFEstimator.from_lead_field(lead_field, np.eye(3))
    with pytest.raises(TypeError, match="Invalid noise type"):
        NCRFEstimator.from_lead_field(lead_field, 'noise-cov.fif')


def test_fit_trims_forward_to_data():
    """Data on a subset of the forward model's channels is fit with a matching sub-forward."""
    rng = np.random.RandomState(0)
    lead_field = NDVar(rng.normal(size=(3, 4)), (SENSOR, Scalar('source', range(4))))
    names = list(SENSOR.names)
    a = rng.normal(size=(3, 3))
    noise = mne.Covariance(a.dot(a.T), names, [], [], 0)
    estimator = NCRFEstimator.from_lead_field(lead_field, noise)

    time = UTS(0, 0.01, 200)
    sensor_sub = Sensor([[0., 0, 1], [1., 0, 0]], ['c', 'a'])
    meg = [NDVar(rng.normal(size=(2, 200)), (sensor_sub, time))]
    stim = [[NDVar(rng.normal(size=200), (time,), name='x')]]
    data = RegressionData.from_data(meg, stim, 0, 0.05, scale=None, stim_is_single=True)

    result = estimator.fit(data, _ShapedZeroSolver())

    # the model's forward is derived for the data's channels, in the data's order
    model = result.model
    assert list(model.forward.sensor.names) == ['c', 'a']
    np.testing.assert_array_equal(model.forward.lead_field, lead_field.x[[2, 0]])
    np.testing.assert_array_equal(model.forward.noise_covariance, noise.data[np.ix_([2, 0], [2, 0])])
    # the estimator itself is unchanged
    assert estimator.forward is None

    # the derived forward is identical to one built from the trimmed inputs directly
    reference = NCRFEstimator.from_lead_field(lead_field.sub(sensor=['c', 'a']), mne.Covariance(noise.data[np.ix_([2, 0], [2, 0])], ['c', 'a'], [], [], 0))._forward_for(sensor_sub)
    np.testing.assert_array_equal(model.forward.whitening_filter, reference.whitening_filter)
    np.testing.assert_array_equal(model.forward.whitened_lead_field, reference.whitened_lead_field)

    # data with channels the lead field does not cover is an error naming the lead field
    sensor_extra = Sensor([[1., 0, 0], [0, 0.5, 0.5]], ['a', 'z'])
    meg_extra = [NDVar(rng.normal(size=(2, 200)), (sensor_extra, time))]
    data_extra = RegressionData.from_data(meg_extra, stim, 0, 0.05, scale=None, stim_is_single=True)
    with pytest.raises(ValueError, match=r"data channels missing from the lead field: \['z'\]"):
        estimator.fit(data_extra, _ShapedZeroSolver())

    # a channel with a lead field but no noise estimate blames the noise covariance
    estimator_sub = NCRFEstimator.from_lead_field(lead_field, mne.Covariance(noise.data[:2, :2], names[:2], [], [], 0))
    meg_full = [NDVar(rng.normal(size=(3, 200)), (SENSOR, time))]
    data_full = RegressionData.from_data(meg_full, stim, 0, 0.05, scale=None, stim_is_single=True)
    with pytest.raises(ValueError, match=r"data channels missing from the noise covariance: \['c'\]"):
        estimator_sub.fit(data_full, _ShapedZeroSolver())


def test_h_scaled():
    """h_scaled restores the original stimulus scale, whichever scaling was used."""
    for scale in [None, 'l1', 'l2', 'spectral']:
        data = _synthetic_data(scale)
        model = _model(data.design, data.design.n_coefficients)
        h, h_scaled = model.h, model.h_scaled
        if scale is None:
            assert h_scaled is h
            continue
        for i, factor in enumerate(data.design.stim_scaling):
            np.testing.assert_allclose(h_scaled[i].x, h[i].x / factor)


def test_h_scaled_matches_unscaled_fit():
    """h_scaled equals the h of an equivalent model whose covariates were not scaled."""
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


@pytest.mark.parametrize('tstop, n_atoms', [(0.02, 1), (0.05, 4)])
def test_h_with_narrow_basis(tstop, n_atoms):
    """A predictor with a single basis function still expands into a full TRF."""
    rng = np.random.RandomState(0)
    time = UTS(0, 0.01, 200)
    meg = [NDVar(rng.normal(size=(3, 200)), (SENSOR, time))]
    bands = Categorial('band', ['low', 'high'])
    n_lags = int(round(tstop / 0.01)) + 1
    for stim, dimnames, shape in [
        (NDVar(rng.normal(size=200), (time,), name='x'), ('source', 'time'), (4, n_lags)),
        (NDVar(rng.normal(size=(2, 200)), (bands, time), name='x'), ('band', 'source', 'time'), (2, 4, n_lags)),
    ]:
        data = RegressionData.from_data(meg, [[stim]], 0, tstop, scale=None, stim_is_single=True)
        assert data.design.basis[0].shape[1] == n_atoms
        model = _model(data.design, data.design.n_coefficients)

        h = model.h

        assert h.dimnames == dimnames
        assert h.shape == shape


def test_gaussian_basis():
    basis = gaussian_basis(4, np.linspace(0, 1, 11), 0.1)
    shifted_basis = gaussian_basis(4, np.linspace(10, 11, 11), 0.1)

    assert basis.shape == (11, 4)
    np.testing.assert_allclose(basis, shifted_basis)


def test_covariate_from_stim():
    stim = load('stim')[0]
    # Test if difference between list of stimuli and concatenated stimuli
    diff = stim.diff('time')

    start = [-20, -20]
    stop = [20, 20]
    filter_lengths = np.subtract(stop, start) + 1
    covariates = covariate_from_stim([stim, diff], filter_lengths, start)

    conc = concatenate([stim, diff.clip(0)], Categorial('rep', ['on', 'off']))
    covariates_conc = covariate_from_stim([conc], filter_lengths, start)

    assert np.array(covariates).shape == np.array(covariates_conc).shape
    np.testing.assert_allclose(np.array(covariates)[0, 0, 0], np.array(covariates_conc)[0, 0, 0], rtol=0.001)

    # Test if shifted covariate array is equal to unshifted
    start = [-20]
    stop = [20]
    filter_lengths = np.subtract(stop, start) + 1
    covariates = covariate_from_stim([stim], filter_lengths, start)

    start = [0]
    stop = [40]
    filter_lengths = np.subtract(stop, start) + 1
    covariates_shift = covariate_from_stim([stim], filter_lengths, start)

    assert covariates[0].shape[0] == len(stim.get_dim('time'))
    np.testing.assert_array_equal(covariates[0][:-20], covariates_shift[0][20:])
