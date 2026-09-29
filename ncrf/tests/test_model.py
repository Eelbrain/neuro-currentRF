"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""
# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
from dataclasses import dataclass, replace
import pickle
from unittest.mock import MagicMock, Mock

import mne
import numpy as np
import pytest

from ncrf import CrossValidation, CVResult, ForwardModel, NCRF, NCRFEstimator, RegressionData, Solver, SolverFit, TRFDesign, fit_model
from ncrf._data import covariate_from_stim
from ncrf._linalg import gaussian_basis
from ncrf._metrics import merge_scores
from ncrf._model import _score_fit
from ncrf._typing import FloatArray
from .fetch import load

from eelbrain import Categorial, NDVar, Scalar, Sensor, UTS, concatenate


SENSOR = Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'c'])


def test_fit_model():
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
    searching_solver.search.assert_called_once_with(data, cv)


def test_default_search_contract():
    """A solver implementing only solve() is a fixed configuration that needs no CV."""
    solver = _ZeroSolver()

    assert solver.without_history() is solver
    # no estimator or cv config is touched, since nothing is cross-validated
    assert solver.search(None, None) == (solver, [])

    # a generic table renders from whatever score keys are present
    cv_results = [
        CVResult(_ZeroSolver(), {'l2_error': 3.0, 'explained_variance': 0.1}),
        CVResult(solver, {'l2_error': 1.0, 'explained_variance': 0.4}),
    ]
    assert 'l2_error' in str(solver.cv_table(cv_results))


def test_score_fit():
    """Training and cross-validation fold scores use the same composition."""
    model = Mock()
    model.evaluate.return_value = {'l2_error': 3.0, 'explained_variance': 0.5}
    solver_fit = Mock()
    solver_fit.score.return_value = {'cross_fit': 1.0}
    data = Mock(forward=object())

    scores = _score_fit(model, solver_fit, data)

    model.evaluate.assert_called_once_with(data)
    solver_fit.score.assert_called_once_with(data.forward, data)
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
    raw = _synthetic_data()
    forward = _forward(noise_covariance=np.eye(3) / 4)
    data = raw.whiten(forward)
    assert data.forward is forward
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


def test_whiten_shares_covariates():
    """Whitening only changes the responses, so arrays derived from the covariates carry over."""
    raw = _synthetic_data('spectral')
    list(raw)  # populate the responses and covariates
    ete = raw.EtE

    data = raw.whiten(_forward())

    assert data.covariates is raw.covariates
    assert data.EtE is ete
    assert 'responses' not in data.__dict__


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
    mask = np.zeros(len(data.samples), dtype=bool)
    mask[10:50] = True

    by_mask = data.timeslice(mask)
    by_index = data.timeslice(np.flatnonzero(mask))

    np.testing.assert_array_equal(by_mask.samples, by_index.samples)
    np.testing.assert_array_equal(by_mask.responses[0], by_index.responses[0])
    np.testing.assert_array_equal(by_mask.covariates[0], by_index.covariates[0])
    assert by_mask.norm_factor == by_index.norm_factor


def test_timeslice_rescales():
    """A slice's arrays are the full dataset's rows, rescaled to the slice's own norm factor."""
    data = _synthetic_data('l2').whiten(_forward())
    idx = np.arange(10, 50)

    fold = data.timeslice(idx)

    assert fold.is_whitened
    # the arrays are built once, on the source dataset, and sliced from there
    assert 'responses' in data.__dict__ and 'covariates' in data.__dict__
    assert 'responses' in fold.__dict__ and 'covariates' in fold.__dict__
    mul = data.norm_factor / fold.norm_factor
    np.testing.assert_allclose(fold.responses[0], data.responses[0][:, idx] * mul)
    np.testing.assert_allclose(fold.covariates[0], data.covariates[0][idx] * mul)
    # and equal the arrays built from the NDVars for the same samples
    rebuilt = replace(fold, samples=fold.samples)
    np.testing.assert_allclose(rebuilt.responses[0], fold.responses[0])
    np.testing.assert_allclose(rebuilt.covariates[0], fold.covariates[0])


def test_pickle_ships_raw_inputs():
    """Pickles carry the inputs and design; the solver's arrays are rebuilt on demand."""
    data = _synthetic_data('l2').whiten(_forward())
    responses, covariates = data.responses, data.covariates  # populate the caches

    restored = pickle.loads(pickle.dumps(data))

    assert 'responses' not in restored.__dict__ and 'covariates' not in restored.__dict__
    np.testing.assert_array_equal(restored.samples, data.samples)
    np.testing.assert_array_equal(restored.forward.whitening_filter, data.forward.whitening_filter)
    np.testing.assert_allclose(restored.responses[0], responses[0])
    np.testing.assert_allclose(restored.covariates[0], covariates[0])


def test_rejects_inconsistent_inputs():
    """Directly constructed datasets have to match their design."""
    data = _synthetic_data()
    meg, stim = data.meg[0], data.stim[0]

    with pytest.raises(ValueError, match="1 MEG segment but 2 stimulus lists"):
        replace(data, stim=[stim, stim])
    with pytest.raises(ValueError, match="segment 1: combining data segments with different sensor"):
        replace(data, meg=[meg, meg.sub(sensor=['a', 'b'])], stim=[stim, stim])
    with pytest.raises(ValueError, match="segment 0: stim dimensions"):
        replace(data, stim=[stim[:1]])
    with pytest.raises(ValueError, match="time axis incompatible with meg"):
        replace(data, stim=[[stim[0], NDVar(stim[1].x, (UTS(1, 0.01, 200),), name='quiet')]])
    with pytest.raises(ValueError, match="does not match design.tstep"):
        replace(data, design=replace(data.design, tstep=0.02))
    with pytest.raises(ValueError, match="samples out of range"):
        replace(data, samples=[0, 200])
    with pytest.raises(ValueError, match="samples is empty"):
        replace(data, samples=[])


def _forward(seed: int = 1, noise_covariance: FloatArray | None = None) -> ForwardModel:
    """Forward model for the sensors of :func:`_synthetic_data`, with 4 sources."""
    rng = np.random.RandomState(seed)
    if noise_covariance is None:
        noise_covariance = np.eye(3)
    return ForwardModel(rng.normal(size=(3, 4)), noise_covariance, Scalar('source', range(4)), SENSOR, None)


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


@pytest.mark.parametrize('scale', ['l1', 'l2', 'spectral'])
def test_normalize_derives_scale(scale):
    """normalize(scale) derives the values from the dataset itself, which is what from_data uses."""
    expected = _synthetic_data(scale)
    raw = _synthetic_data()

    derived = raw.normalize(scale)

    assert derived.design.scale == scale
    np.testing.assert_array_equal(derived.design.stim_baseline, expected.design.stim_baseline)
    np.testing.assert_array_equal(derived.design.stim_scaling, expected.design.stim_scaling)
    np.testing.assert_allclose(derived.covariates[0], expected.covariates[0])
    # the handed-over 'spectral' covariates equal ones rebuilt from the predictors
    rebuilt = replace(derived, design=derived.design)
    assert 'covariates' not in rebuilt.__dict__
    np.testing.assert_allclose(rebuilt.covariates[0], derived.covariates[0])
    with pytest.raises(ValueError, match="need one of"):
        raw.normalize('l3')


def test_normalize_replaces_normalization():
    """Covariates are built from the raw predictors, so any normalization can be swapped for another."""
    spectral = _synthetic_data('spectral')
    l2 = _synthetic_data('l2')

    renormalized = spectral.normalize(l2.design)

    assert renormalized.design is l2.design
    np.testing.assert_allclose(renormalized.covariates[0], l2.covariates[0])
    # the source dataset is unchanged
    assert spectral.design.scale == 'spectral'
    # applying the same normalization again is a no-op
    np.testing.assert_array_equal(spectral.normalize(spectral.design).covariates[0], spectral.covariates[0])


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


def test_from_data_rejects_flat_channels():
    """A flat channel makes the noise covariance rank deficient, so it is rejected up front."""
    rng = np.random.RandomState(0)
    time = UTS(0, 0.01, 200)
    flat = rng.normal(size=(3, 200))
    flat[1] = 0
    meg = [NDVar(rng.normal(size=(3, 200)), (SENSOR, time)), NDVar(flat, (SENSOR, time))]
    stim = [[NDVar(rng.normal(size=200), (time,), name='x')] for _ in meg]

    with pytest.raises(ValueError, match=r"segment 1 has flat channels \(b\)"):
        RegressionData.from_data(meg, stim, 0, 0.05)


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
    for expected, actual in zip(predicted, model.predict(data.whiten(forward))):
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
    meg = data.meg[0]
    # same channels in a different order: whitening would apply to the wrong channels
    reordered = replace(data, meg=[NDVar(meg.x, (Sensor([[0., 0, 1], [0, 1, 0], [1, 0, 0]], ['c', 'b', 'a']), meg.time))])

    model = _model(data.design, data.design.n_coefficients)
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.predict(reordered)
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.evaluate(reordered)

    # a genuinely different channel set names the channels that differ
    renamed = replace(data, meg=[NDVar(meg.x, (Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'z']), meg.time))])
    with pytest.raises(ValueError, match=r"only in data: \['z'\]; only in forward model: \['c'\]"):
        model.predict(renamed)

    with pytest.raises(ValueError, match="no lead field to derive a forward model from"):
        NCRFEstimator().fit(renamed, _ZeroSolver())


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

    # data whitened with another model's forward is not silently fit with that
    # lead field instead of the estimator's
    other_forward = ForwardModel(rng.normal(size=(2, 4)), noise.data[np.ix_([2, 0], [2, 0])], Scalar('source', range(4)), sensor_sub, None)
    with pytest.raises(ValueError, match="whitened with a different forward model"):
        estimator.fit(data.whiten(other_forward), _ShapedZeroSolver())
    assert estimator.fit(data.whiten(model.forward), _ShapedZeroSolver()).model.forward is model.forward

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


def test_design_timing_per_predictor():
    """A scalar TRF time applies to all predictors, a sequence has one value per predictor."""
    time = UTS(0, 0.01, 200)
    stim = [NDVar(np.zeros(200), (time,), name='a'), NDVar(np.zeros(200), (time,), name='b')]

    design = TRFDesign.from_stim(stim, 0.01, -0.1, [0.2, 0.3])
    assert design.tstart == [-0.1, -0.1]
    assert design.tstop == [0.2, 0.3]
    # a one-element sequence is not broadcast
    with pytest.raises(ValueError, match="need one value per predictor"):
        TRFDesign.from_stim(stim, 0.01, [-0.1], 0.2)


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
