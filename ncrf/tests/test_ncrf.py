# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
import pickle
import numpy as np

from ncrf import fit_ncrf
from ncrf.tests.fetch import load

from eelbrain import Categorial, concatenate
from eelbrain.testing import assert_dataobj_equal


def test_ncrf():
    meg = load('meg').sub(time=(0, 5))
    stim = load('stim').sub(time=(0, 5))
    fwd = load('fwd_sol')
    emptyroom = load('emptyroom')

    # 1 stimulus
    model = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu=0.0019444, n_iter=3, n_iterc=3,
                     n_iterf=10, do_post_normalization=False)
    # check reduce
    model.reduce_data()
    assert model._data is None
    assert getattr(model, "megmeta", None) is not None
    # check reconstruct
    model.reconstruct_data(meg, stim, attach=True)
    assert model._data is not None

    # check residual and explained var
    np.testing.assert_allclose(model.explained_var, 0.00641890144769941, rtol=0.001)
    np.testing.assert_allclose(model.voxelwise_explained_variance.sum(), 0.08261162457414245, rtol=0.001)
    np.testing.assert_allclose(model.residual, 178.512, rtol=0.001)
    # check scaling
    stim_baseline = stim.mean()
    np.testing.assert_equal(model._stim_baseline[0], stim_baseline)
    np.testing.assert_equal(model._stim_scaling[0], (stim - stim_baseline).abs().mean())
    np.testing.assert_allclose(model.h.norm('time').norm('source').norm('space'), 6.601677e-10, rtol=0.001)

    # test persistence
    model_2 = pickle.loads(pickle.dumps(model, pickle.HIGHEST_PROTOCOL))
    assert_dataobj_equal(model_2.h, model.h)
    assert_dataobj_equal(model_2.h_scaled, model.h_scaled)
    np.testing.assert_equal(model_2.residual, model.residual)
    np.testing.assert_equal(model_2.gaussian_fwhm, model.gaussian_fwhm)

    # test gaussian fwhm
    model = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu=0.0019444, n_iter=1, n_iterc=1,
                     n_iterf=1, gaussian_fwhm=50.0)
    assert model.gaussian_fwhm == 50.0

    # 2 stimuli, one of them 2-d, normalize='l2'
    diff = stim.diff('time')
    stim2 = concatenate([diff.clip(0), diff.clip(max=0)], Categorial('rep', ['on', 'off']))
    model = fit_ncrf(meg, [stim, stim2], fwd, emptyroom, tstop=[0.2, 0.2], normalize='l2', mu=0.0019444, n_iter=3,
                     n_iterc=3, n_iterf=10, do_post_normalization=False)
    # check scaling
    np.testing.assert_equal(model._stim_baseline[0], stim.mean())
    np.testing.assert_equal(model._stim_scaling[0], stim.std())
    np.testing.assert_allclose(model.h[0].norm('time').norm('source').norm('space'), 7.0088e-10, rtol=0.001)

    # 2 stimuli, different tstarts (-ve)
    diff = stim.diff('time')
    stim2 = concatenate([diff.clip(0), diff.clip(max=0)], Categorial('rep', ['on', 'off']))
    tstart = [-0.1, 0.1]
    tstop = [0.2, 0.3]
    model = fit_ncrf(meg, [stim, stim2], fwd, emptyroom, tstart=tstart, tstop=tstop, normalize='l2', mu=0.0019444, n_iter=3,
                     n_iterc=3, n_iterf=10, do_post_normalization=False)

    # check residual and explained var
    np.testing.assert_allclose(model.explained_var, 0.02148945636352262, rtol=0.001)
    np.testing.assert_allclose(model.residual, 176.953, rtol=0.001)
    # check start and stop
    np.testing.assert_equal(model.tstart, tstart)
    np.testing.assert_equal(model.tstop, tstop)
    # check scaling
    np.testing.assert_equal(model._stim_baseline[0], stim.mean())
    np.testing.assert_equal(model._stim_scaling[0], stim.std())
    np.testing.assert_allclose(model.h[0].norm('time').norm('source').norm('space'), 5.9598e-10, rtol=0.001)

    # cross-validation
    model = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu='auto', n_iter=1, n_iterc=2, n_iterf=2, do_post_normalization=False)
    np.testing.assert_allclose(model.mu, 0.0203, rtol=0.001)
    model.cv_info()

    # test without multiprocessing
    model_no_mp = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu='auto', n_iter=1, n_iterc=2, n_iterf=2, n_workers=0, do_post_normalization=False)
    assert_dataobj_equal(model_no_mp.h, model.h)
