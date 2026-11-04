"""Tests for the Gaussian basis over the source space."""

# License: BSD (3-clause)
import logging
import pickle

from eelbrain.testing import assert_dataobj_equal
import numpy as np
import pytest

from ncrf import ForwardModel, NCRFEstimator, RegressionData, SourceBasis, fit_ncrf
from ncrf._source_basis import _SUPPORT
from ncrf.tests.fetch import load, surface_source_space

#: Kernel width small enough that every atom covers only its own center, i.e. the basis is the identity.
TINY_STD = 1e-5


@pytest.fixture(scope='module')
def lead_field():
    return load('fwd_sol', geometry=True)


@pytest.fixture(scope='module')
def volume(lead_field):
    return lead_field.get_dim('source')


@pytest.fixture(scope='module')
def surface():
    return surface_source_space()


def _grid_index(volume):
    positions = volume.coordinates
    return np.round((positions - positions.min(0)) / (volume.grade / 1000)).astype(int)


@pytest.mark.parametrize('stride', [1, 2])
def test_volume_basis(volume, stride):
    std = 0.006
    basis = SourceBasis.from_source(volume, std, stride)
    phi = basis.basis
    positions = volume.coordinates

    expected_centers = np.flatnonzero((_grid_index(volume) % stride == 0).all(1))
    np.testing.assert_array_equal(basis.centers, expected_centers)
    assert phi.shape == (len(volume), len(expected_centers))
    assert (basis.stride, basis.std) == (stride, std)
    # each atom peaks at 1 on its center
    np.testing.assert_array_equal(phi[basis.centers, np.arange(basis.n_atoms)], 1)
    assert phi.max() == 1
    # Gaussian in the Euclidean distance, within the kernel support
    phi = phi.tocoo()
    distance = np.linalg.norm(positions[phi.row] - positions[basis.centers[phi.col]], axis=1)
    np.testing.assert_allclose(phi.data, np.exp(-distance ** 2 / (2 * std ** 2)))
    assert distance.max() <= _SUPPORT * std
    # truncated at the support, not earlier: a dense kernel misses only values below 1e-6
    dense = np.exp(-np.linalg.norm(positions[:, None] - positions[None, basis.centers], axis=-1) ** 2 / (2 * std ** 2))
    dense[positions[:, None, 0] * positions[None, basis.centers, 0] < 0] = 0
    assert np.abs(dense - phi.toarray()).max() < 1e-6
    # no atom extends across the midline
    assert (positions[phi.row, 0] * positions[basis.centers[phi.col], 0] > 0).all()


def test_surface_basis(surface):
    std = 0.006
    basis = SourceBasis.from_source(surface, std)
    phi = basis.basis
    n_lh = len(surface.vertices[0])

    assert phi.shape == (len(surface), len(surface))
    np.testing.assert_array_equal(basis.centers, np.arange(len(surface)))
    np.testing.assert_array_equal(phi.diagonal(), 1)
    assert phi.max() == 1
    # the hemispheres are not connected
    assert phi[:n_lh, n_lh:].nnz == phi[n_lh:, :n_lh].nnz == 0
    # geodesic distances are at least the Euclidean ones, and longer across sulci
    phi = phi.tocoo()
    positions = surface.coordinates
    euclidean = np.exp(-np.linalg.norm(positions[phi.row] - positions[phi.col], axis=1) ** 2 / (2 * std ** 2))
    assert (phi.data <= euclidean + 1e-12).all()
    assert (phi.data < euclidean - 0.1).any()
    # symmetric, since geodesic distance is
    assert abs(basis.basis - basis.basis.T).max() < 1e-12


@pytest.mark.parametrize('source, std, stride, error, match', [
    ('volume', 0, 1, ValueError, 'std='),
    ('volume', -0.006, 1, ValueError, 'std='),
    ('volume', np.nan, 1, ValueError, 'std='),
    ('volume', np.inf, 1, ValueError, 'std='),
    # millimeters rather than meters
    ('volume', 6, 1, ValueError, 'larger than the source space'),
    ('surface', 6, 1, ValueError, 'larger than the source space'),
    ('volume', 0.006, 0, ValueError, 'stride='),
    ('volume', 0.006, 1.5, ValueError, 'stride='),
    ('volume', 0.006, True, ValueError, 'stride='),
    ('volume', 0.006, 50, ValueError, 'no atom centers'),
    ('surface', 0.006, 2, NotImplementedError, 'stride=2'),
    # the test source space has geodesic distances up to 40 mm, less than 5.3 * 8 mm
    ('surface', 0.008, 1, ValueError, 'geodesic distances only up to'),
])
def test_invalid(request, source, std, stride, error, match):
    source = request.getfixturevalue(source)
    with pytest.raises(error, match=match):
        SourceBasis.from_source(source, std, stride)


@pytest.mark.parametrize('stride', [0, 2])
def test_fit_ncrf_rejects_stride_without_std(stride):
    # rejected before the inputs are looked at
    with pytest.raises(ValueError, match='without source_basis_std'):
        fit_ncrf(None, None, None, None, source_basis_stride=stride)


def test_spacing_warning(volume, caplog):
    with caplog.at_level(logging.WARNING, 'ncrf._source_basis'):
        SourceBasis.from_source(volume, 0.006, 2)
    assert not caplog.records
    # 14 mm between centers, 3.5 std
    with caplog.at_level(logging.WARNING, 'ncrf._source_basis'):
        SourceBasis.from_source(volume, 0.004, 2)
    assert 'barely overlap' in caplog.text


@pytest.mark.parametrize('dc', [1, 3])
def test_reduce_expand(volume, dc):
    basis = SourceBasis.from_source(volume, 0.006, 2)
    rng = np.random.default_rng(0)
    lead_field = rng.normal(size=(5, basis.n_sources * dc))
    coefficients = rng.normal(size=(basis.n_atoms * dc, 4))
    phi = np.kron(basis.basis.toarray(), np.eye(dc))

    np.testing.assert_allclose(basis.reduce(lead_field, dc), lead_field @ phi)
    np.testing.assert_allclose(basis.expand(coefficients, dc), phi @ coefficients)


def test_equals(volume):
    basis = SourceBasis.from_source(volume, 0.006)
    assert basis.equals(basis)
    assert basis.equals(SourceBasis.from_source(volume, 0.006))
    assert not basis.equals(None)
    assert not basis.equals(SourceBasis.from_source(volume, 0.005))
    assert not basis.equals(SourceBasis.from_source(volume, 0.006, 2))


def _forward(lead_field, source_basis=None):
    estimator = NCRFEstimator.from_lead_field(lead_field, load('emptyroom'), source_basis)
    return estimator._forward_for(lead_field.get_dim('sensor'))


def test_forward_model(lead_field, volume):
    basis = SourceBasis.from_source(volume, 0.006, 2)
    forward = _forward(lead_field)
    forward_basis = _forward(lead_field, basis)
    dc = forward.dc

    assert forward_basis.source_basis is basis
    assert forward_basis.n_components == basis.n_atoms * dc
    assert forward_basis.whitened_lead_field.shape == (forward.rank, basis.n_atoms * dc)
    # the solver's lead field is that of the atoms, normalized by its own spectral norm
    atoms = basis.reduce(forward.whitening_filter @ forward.lead_field, dc)
    assert forward_basis.lead_field_scaling == pytest.approx(np.linalg.norm(atoms, 2))
    np.testing.assert_allclose(forward_basis.whitened_lead_field, atoms / forward_basis.lead_field_scaling)
    # expanded coefficients predict the same through the lead field of the sources
    theta = np.random.default_rng(0).normal(size=(forward_basis.n_components, 2))
    np.testing.assert_allclose(forward.whitening_filter @ forward.lead_field @ forward_basis.expand(theta) / forward_basis.lead_field_scaling, forward_basis.whitened_lead_field @ theta)
    # without a basis, coefficients are already over the sources
    assert forward.expand(theta) is theta
    assert forward.n_components == len(volume) * dc

    # the identity basis reproduces the forward model without a basis
    identity = _forward(lead_field, SourceBasis.from_source(volume, TINY_STD))
    np.testing.assert_allclose(identity.whitened_lead_field, forward.whitened_lead_field)
    assert identity.lead_field_scaling == pytest.approx(forward.lead_field_scaling)

    # the derived quantities are recomputed on unpickling
    forward_2 = pickle.loads(pickle.dumps(forward_basis))
    assert forward_2.source_basis.equals(basis)
    np.testing.assert_array_equal(forward_2.whitened_lead_field, forward_basis.whitened_lead_field)
    # forward models pickled before source bases existed load without one
    state = forward.__getstate__()
    del state['source_basis']
    forward_3 = ForwardModel.__new__(ForwardModel)
    forward_3.__setstate__(state)
    assert forward_3.source_basis is None
    np.testing.assert_array_equal(forward_3.whitened_lead_field, forward.whitened_lead_field)


def test_rejects_mismatched_basis(lead_field, surface):
    basis = SourceBasis.from_source(surface, 0.006)
    with pytest.raises(ValueError, match='source_basis over 1284 sources'):
        NCRFEstimator.from_lead_field(lead_field, load('emptyroom'), basis)
    forward = _forward(lead_field)
    with pytest.raises(ValueError, match='source_basis over 1284 sources'):
        ForwardModel(forward.lead_field, forward.noise_covariance, forward.source, forward.sensor, forward.space, basis)


def test_whiten_swaps_source_basis(lead_field, volume):
    meg = load('meg').sub(time=(0, 1))
    stim = load('stim').sub(time=(0, 1))
    raw = RegressionData.from_data(list(meg), [[x] for x in stim], 0, 0.1, scale='l1')
    forward = _forward(lead_field)
    basis = SourceBasis.from_source(volume, 0.006, 2)
    forward_basis = _forward(lead_field, basis)

    data = raw.whiten(forward)
    data.responses
    data_basis = data.whiten(forward_basis)
    # the solver reads the source basis from the dataset's forward model
    assert data_basis.forward is forward_basis
    # the whitening is the same, so the whitened responses are shared
    assert data_basis.responses is data.responses
    assert data_basis.whiten(_forward(lead_field, SourceBasis.from_source(volume, 0.006, 2))) is data_basis
    assert data_basis.whiten(forward) is not data_basis


@pytest.mark.slow
def test_fit_identity_basis(lead_field, volume):
    """A basis of single-source atoms reproduces the fit without a basis."""
    meg = load('meg').sub(time=(0, 5))
    stim = load('stim').sub(time=(0, 5))
    emptyroom = load('emptyroom')
    kwargs = dict(tstop=0.2, scale='l1', mu=0.0019444, n_iter=2, n_iterc=2, n_iterf=5)

    result = fit_ncrf(meg, stim, lead_field, emptyroom, **kwargs)
    result_identity = fit_ncrf(meg, stim, lead_field, emptyroom, source_basis_std=TINY_STD, **kwargs)

    for key, value in result.scores.items():
        assert result_identity.scores[key] == pytest.approx(value, rel=1e-6)
    np.testing.assert_allclose(result_identity.model.h.x, result.model.h.x, rtol=1e-6, atol=1e-6 * np.abs(result.model.h.x).max())
    np.testing.assert_allclose(result_identity.voxelwise_explained_variance.x, result.voxelwise_explained_variance.x, rtol=1e-6, atol=1e-12)


@pytest.mark.slow
def test_fit_source_basis(lead_field, volume):
    meg = load('meg').sub(time=(0, 5))
    stim = load('stim').sub(time=(0, 5))
    emptyroom = load('emptyroom')

    result = fit_ncrf(meg, stim, lead_field, emptyroom, tstop=0.2, scale='l1', mu=0.0019444, n_iter=2, n_iterc=2, n_iterf=5, source_basis_std=0.006, source_basis_stride=2)
    model = result.model
    forward = model.forward
    basis = forward.source_basis
    assert (basis.std, basis.stride, basis.n_atoms) == (0.006, 2, 464)
    # the coefficients and the Champagne source covariances are over the atoms
    assert model.theta.shape == (basis.n_atoms * 3, model.design.n_coefficients)
    assert all(gamma.shape == (basis.n_atoms, 3, 3) for gamma in result.solver_fit.gamma)
    # the response function and the explained variance are over the sources
    assert model.h.get_dim('source') == volume
    assert result.voxelwise_explained_variance.get_dim('source') == volume
    assert np.isfinite(result.voxelwise_explained_variance.x).all()
    assert np.isfinite(result.scores['explained_variance'])
    # sensor-space prediction through the lead field of the sources agrees with the whitened one
    data = RegressionData.from_data(list(meg), [[x] for x in stim], 0, 0.2, scale=None).normalize(model.design)
    prediction = model.predict(data)
    whitened = model.predict(data, whitened=True)
    for y, y_whitened in zip(prediction, whitened):
        np.testing.assert_allclose(forward.whitening_filter @ y / data.norm_factor, y_whitened, rtol=1e-8, atol=1e-8 * np.abs(y_whitened).max())
    assert 'atoms' in repr(model)

    model_2 = pickle.loads(pickle.dumps(model, pickle.HIGHEST_PROTOCOL))
    assert model_2.forward.source_basis.equals(basis)
    assert_dataobj_equal(model_2.h, model.h)
