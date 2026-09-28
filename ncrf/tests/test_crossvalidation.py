"""Tests for cross-validation orchestration and selection."""

from unittest.mock import Mock

import numpy as np
import pytest

from ncrf import ChampLasso, CrossValidation, crossvalidate
from ncrf import _crossvalidation as cv
from ncrf._solvers import champ_lasso


def _cv_result(mu, *, cross_fit=0.0, es=0.0, l2_error=0.0):
    return cv.CVResult(ChampLasso(mu=mu), {
        'cross_fit': cross_fit,
        'estimation_stability': es,
        'l2_error': l2_error,
        'weighted_l2_error': 0.0,
    })


class _Progress:
    def __init__(self):
        self.updates = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def update(self, n=1):
        self.updates.append(n)


class _InlinePool:
    def __init__(self, *, processes, initializer, initargs):
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def imap_unordered(self, function, values):
        return map(function, values)


def test_make_folds(monkeypatch):
    train_data = object()
    test_data = object()
    data = Mock()
    data.design.filter_length = [2, 5]
    data.samples = np.arange(10)
    data.timeslice.side_effect = [train_data, test_data]

    splitter = Mock()
    splitter.split.return_value = [(np.array([0, 1]), np.array([2]))]
    captured = {}

    def make_splitter(**kwargs):
        captured.update(kwargs)
        return splitter

    monkeypatch.setattr(cv, 'TimeSeriesSplit', make_splitter)

    folds = cv._make_folds(data, 2)

    # the gap must cover the longest TRF, so lagged training predictors
    # cannot reach into the held-out window
    assert captured == {'r': 0.05, 'p': 2, 'd': 5}
    assert folds == [(train_data, test_data)]


def test_score_candidate_uses_estimator_fit_primitive(monkeypatch):
    train_data = object()
    test_data = object()
    data = Mock()
    monkeypatch.setattr(cv, 'compute_es_metric', lambda models, full_data: 4.0)

    model = Mock()
    solver_fit = Mock()
    estimator = Mock()
    solver = ChampLasso(mu=0.1, tol=1e-5)
    estimator.fit_model.return_value = model, solver_fit
    estimator._score_fit.return_value = {'cross_fit': 1.0, 'weighted_l2_error': 2.0, 'l2_error': 3.0, 'explained_variance': 0.5}

    result = cv._score_candidate(estimator, data, [(train_data, test_data)], solver)

    fold_solver = estimator.fit_model.call_args.args[1]
    assert fold_solver.mu == 0.1
    assert fold_solver.tol == solver.tol
    assert not fold_solver.store
    estimator.fit_model.assert_called_once_with(train_data, fold_solver)
    estimator._score_fit.assert_called_once_with(model, solver_fit, test_data)
    assert result.solver is solver
    assert result.scores == {
        'cross_fit': 1.0,
        'weighted_l2_error': 2.0,
        'l2_error': 3.0,
        'explained_variance': 0.5,
        'estimation_stability': 4.0,
    }


def test_time_series_split_rejects_empty_training_window():
    splitter = cv.TimeSeriesSplit(r=0.05, p=3, d=100)

    train, test = next(splitter.split(np.empty(400)))
    assert (len(train), len(test)) == (240, 20)

    with pytest.raises(ValueError, match="110 samples are not enough for 3 cross-validation folds"):
        list(splitter.split(np.empty(110)))


def test_cross_validation_rejects_no_folds():
    with pytest.raises(ValueError, match="n_splits=0: need at least one cross-validation fold"):
        CrossValidation(n_splits=0)


def test_extend_mu_grid():
    champ = ChampLasso(mu=0.1)
    mus = (0.1, 0.2, 0.3)

    left = champ._extend_grid([_cv_result(mu, cross_fit=mu) for mu in mus])
    right = champ._extend_grid([_cv_result(mu, cross_fit=-mu) for mu in mus])
    interior = champ._extend_grid([_cv_result(mu, cross_fit=abs(mu - 0.2)) for mu in mus])

    np.testing.assert_allclose([solver.mu for solver in left], np.logspace(-2, -1, 4)[:-1])
    np.testing.assert_allclose([solver.mu for solver in right], np.logspace(np.log10(0.3), np.log10(3), 4)[1:])
    assert interior == ()


def test_select_es_solver():
    results = [
        _cv_result(0.1, es=5.0),
        _cv_result(0.2, es=4.0),
        _cv_result(0.3, es=2.0),
        _cv_result(0.4, es=3.0),
    ]

    assert champ_lasso._select_es_solver(results, 0.2).mu == 0.3
    assert champ_lasso._select_es_solver(results, 0.4) is None


def test_crossvalidate_progress(monkeypatch):
    progress = _Progress()
    monkeypatch.setattr(cv, 'tqdm', lambda **kwargs: progress)
    monkeypatch.setattr(cv, '_make_folds', lambda data, n_splits: [])
    monkeypatch.setattr(
        cv, '_score_candidate',
        lambda estimator, data, folds, solver: _cv_result(solver.mu),
    )
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2, 0.3))

    results = crossvalidate(
        object(), object(), candidates, CrossValidation(n_splits=2, n_workers=0),
    )

    assert [result.solver.mu for result in results] == [0.1, 0.2, 0.3]
    assert progress.updates == [1, 1, 1]
    assert progress.closed


def test_crossvalidate_propagates_worker_error(monkeypatch):
    progress = _Progress()
    monkeypatch.setattr(cv, 'tqdm', lambda **kwargs: progress)
    monkeypatch.setattr(cv, 'Pool', _InlinePool)
    monkeypatch.setattr(cv, '_make_folds', lambda data, n_splits: [])

    def score_candidate(estimator, data, folds, solver):
        if solver.mu == 0.2:
            raise RuntimeError("worker failed")
        return _cv_result(solver.mu)

    monkeypatch.setattr(cv, '_score_candidate', score_candidate)
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2))

    with pytest.raises(RuntimeError, match="worker failed"):
        crossvalidate(
            object(), object(), candidates, CrossValidation(n_splits=2, n_workers=2),
        )

    assert progress.updates == [1]
    assert progress.closed


def test_search_single_candidate_skips_crossvalidation(monkeypatch):
    crossvalidate = Mock()
    monkeypatch.setattr(champ_lasso, 'crossvalidate', crossvalidate)

    solver, cv_results = ChampLasso(mu=0.1).search(Mock(), None, None)

    assert (solver.mu, cv_results) == (0.1, [])
    crossvalidate.assert_not_called()


def test_search_extends_grid_before_es_selection(monkeypatch):
    """The boundary extension follows the cross-fit winner, not the ES selection."""
    mus = (0.1, 0.2, 0.3, 0.4)
    extension = [float(mu) for mu in np.logspace(-2, -1, 4)[:-1]]
    # cross-fit selects the smallest mu, while ES would select an interior candidate
    scores = {  # mu: (cross_fit, estimation_stability)
        0.1: (1.0, 5.0),
        0.2: (2.0, 4.0),
        0.3: (3.0, 2.0),
        0.4: (4.0, 3.0),
        extension[0]: (3.0, 9.0),
        extension[1]: (0.5, 1.0),
        extension[2]: (0.8, 0.5),
    }
    calls = []
    estimator = Mock()
    cv_config = CrossValidation(n_splits=2, n_workers=0)

    def crossvalidate(called_estimator, data, candidates, called_cv):
        assert (called_estimator, called_cv) == (estimator, cv_config)
        calls.append([solver.mu for solver in candidates])
        return [_cv_result(solver.mu, cross_fit=scores[solver.mu][0], es=scores[solver.mu][1]) for solver in candidates]

    monkeypatch.setattr(champ_lasso, 'crossvalidate', crossvalidate)

    solver, returned_results = ChampLasso(mu=mus, use_es=True).search(estimator, None, cv_config)

    assert calls == [list(mus), extension]
    # ES minimum above the extended cross-fit winner (extension[1]), not the 0.3 of the truncated grid
    assert solver.mu == extension[2]
    assert [result.solver.mu for result in returned_results] == [*mus, *extension]


def test_search_es_is_independent_of_result_order(monkeypatch):
    results = [
        _cv_result(0.1, es=5.0, cross_fit=3.0),
        _cv_result(0.3, es=2.0, cross_fit=2.0),
        _cv_result(0.4, es=3.0, cross_fit=4.0),
        _cv_result(0.2, es=4.0, cross_fit=1.0),
    ]
    monkeypatch.setattr(champ_lasso, 'crossvalidate', lambda *args: results.copy())

    solver, returned_results = ChampLasso(mu=(0.1, 0.2, 0.3, 0.4), use_es=True).search(Mock(), None, None)

    assert solver.mu == 0.3
    assert returned_results == results
    assert '0.30000*' in str(solver.cv_table(returned_results))
