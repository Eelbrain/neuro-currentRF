# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Christian Brodbeck <email:brodbecc@mcmaster.ca>
#          Marlies Gillis <email: >
# License: BSD (3-clause)
from typing import List
import collections

from eelbrain import NDVar, Sensor
from mne import Covariance
import numpy as np

from ._model import NCRF, RegressionData


DEFAULT_MUs = np.logspace(-3, -1, 7)


def _handle_noise_channels(noise: [NDVar, Covariance, np.ndarray], sensor_dim: Sensor) -> np.ndarray:

    if isinstance(noise, NDVar):
        er = noise.get_data(('sensor', 'time'))
        noise_cov = np.dot(er, er.T) / er.shape[1]

    elif isinstance(noise, Covariance):
        chs_noise = set(noise.ch_names)
        chs_data = set(sensor_dim.names)
        missing = sorted(chs_data - chs_noise)
        if missing:
            raise RuntimeError(f"Missing channels in noise covariance: {', '.join(missing)}")

        index = [noise.ch_names.index(ch) for ch in sensor_dim.names]

        if noise['diag']:
            full_cov = np.zeros((len(noise.data), len(noise.data)))
            row, col = np.diag_indices(full_cov.shape[0])
            full_cov[row, col] = noise.data
            noise_cov = full_cov[index, :][:, index]
        else:
            noise_cov = noise.data[index, :][:, index]

    elif isinstance(noise, np.ndarray):
        n = len(sensor_dim)
        if noise.shape != (n, n):
            raise ValueError(f"noise = array of shape {noise.shape}; should be {(n, n)}")
        noise_cov = noise

    else:
        raise TypeError(f"Invalid noise type: {type(noise)}. Must be NDVar, Covariance, or ndarray.")

    return noise_cov


def fit_ncrf(meg, stim, lead_field, noise, tstart=0, tstop=0.5, nlevels=1,
             n_iter=10, n_iterc=10, n_iterf=100, normalize=False, in_place=False,
             mu='auto', tol=1e-3, verbose=False, n_splits=3, n_workers=None,
             use_ES=False, gaussian_fwhm=20.0, do_post_normalization=True):
    r"""One shot function for cortical TRF localization.

    Estimate both TRFs and source variance from the observed MEG data by solving
    the Bayesian optimization problem mentioned in :cite:p:`das2020neuro`.

    Parameters
    ----------
    meg :  NDVar ([case,] sensor, time) | list[NDVar]
        If multiple trials are the same length they can be specified as
        :class:`eelbrain.NDVar` with case dimension, if they are different length they
        can be supplied as list.
    stim : eelbrain.NDVar ([case, dim,] time) | list[NDVar]
        One or multiple predictors corresponding to each item in ``meg``.
    lead_field : NDVar
        forward solution a.k.a. lead field matrix.
    noise : mne.Covariance | NDVar | np.ndarray
        The empty room noise covariance, or data from which to compute it as
        :class:`eelbrain.NDVar`.
    tstart : float | list[float]
        Start of the TRF in seconds. Can define multiple tstarts for more than 1 predictor.
    tstop : float | list[float]
        Stop of the TRF in seconds. Can define multiple tstops for more than 1 predictor.
    nlevels : int
        Decides the density of Gabor atoms. Bigger nlevel -> less dense basis.
        By default it is set to `1`. `nlevesl > 2` should be used with caution.
    n_iter : int
        Number of out iterations of the algorithm, by default set to 10.
    n_iterc : int
        Number of Champagne iterations within each outer iteration, by default set to 10.
    n_iterf : int
        Number of FASTA iterations within each outer iteration, by default set to 100.
    normalize : bool | 'l2' | 'l1'
        Scale ``stim`` before model fitting: subtract the mean and divide by
        the standard deviation (when ``nomrmalize='l2'`` or ``normalize=True``)
        or the mean absolute value (when ``normalize='l1'``). By default,
        ``normalize=False`` leaves ``stim`` data untouched.
    in_place: bool
        With ``in_place=False`` (default) the original ``meg`` and ``stims`` are left untouched;
        use ``in_place=True`` to save memory by using the original ``meg`` and ``stim``.
    mu : 'auto' | float | sequence[float]
        Choice of regularizer parameters. Specify a single value to fit a model
        corresponding to that value. Alternatively, specify a range over which
        cross-validation will be done. By default (``mu='auto'``) a range of
        values for cross-validation is chosen based on the data.
    tol : float
        Tolerance factor deciding stopping criterion for the overall algorithm. The iterations
        are stooped when ``norm(trf_new - trf_old)/norm(trf_old) < tol`` condition is met.
        By default ``tol=1e-3``.
    verbose : bool
        If True prints intermediate results, by default False.
    n_splits : int
        Number of cross-validation folds. By default it uses 3-fold cross-validation.
    n_workers : int (optional)
        Number of workers to spawn for cross-validation. If None, it will use ``cpu_count/2``.
    use_ES : bool (optional)
        Use estimation stability criterion :cite:`limEstimationStabilityCrossValidation2016` to
        choose the best ``mu``. (False, by default)
    gaussian_fwhm : float (optional)
        Specifies the full width half maximum (fwmh) for the Gaussian kernel (used as elements of
        the time basis), the default is 20 ms. The standard deviation (std) is related to the
        fwmh as following:
        :math:`std = fwhm / (2 * (sqrt(2 * log(2))))`.
    do_post_normalization : bool (optional)
        Scales covariate matrices of different predictor variables by spectral norms to
        equalize their spectral spread (=1). (True, by default)

    Returns
    -------
    trf : NCRF
        The result of the model fit.

    Examples
    --------
    MEG data ``y`` with dimensions (case, sensor, time) and predictor ``x``
    with dimensions (case, time)::

        ncrf(y, x, fwd, cov)

    ``x`` can always also have an additional predictor dimension, for example,
    if ``x`` represents a spectrogram: (case, frequency, time). The case
    dimension is optional, i.e. a single contiguous data segment is also
    accepted, but the case dimension should always match between ``y`` and
    ``x``.

    Multiple distinct predictor variables can be supplied as list; e.g., when
    modeling simultaneous responses to an attended and an unattended stimulus
    with ``x_attended`` and ``x_unattended``::

        ncrf(y, [x_attended, x_unattended], fwd, cov)

    Multiple data segments can also be specified as list. E.g., if ``y1`` and
    ``y2`` are responses to stimuli ``x1`` and ``x2``, respoectively::

        ncrf([y1, y2], [x1, x2], fwd, cov)

    And with multiple predictors::

        ncrf([y1, y2], [[x1_attended, x1_unattended], [x2_attended, x2_unattended]], fwd, cov)

    References
    ----------
    .. bibliography::
        :cited:
    """
    # data copy?
    if not isinstance(in_place, bool):
        raise TypeError(f"in_place={in_place!r}, need bool")

    # make meg/stim representation uniform:
    meg_trials = []  # [trial_1, trial_2, ...]
    stim_trials = []  # [[trial_1_stim_1, trial_1_stim_2, ...], ...]
    if isinstance(meg, NDVar):
        meg_list = [meg]
        stim_list = [stim]
    elif isinstance(meg, collections.abc.Sequence):
        if len(stim) != len(meg):
            raise ValueError(f"meg={meg}, stim={stim}: different length")
        meg_list = list(meg)
        stim_list = list(stim)
    else:
        raise TypeError(f"meg={meg!r}")
    stim_is_single = None
    for meg_chunk, stim_chunk in zip(meg_list, stim_list):
        if meg_chunk.has_case:
            n_trials = len(meg_chunk)
            meg_trials.extend(meg_chunk)
        else:
            n_trials = 0
            meg_trials.append(meg_chunk)

        if stim_is_single is None:
            stim_is_single = isinstance(stim_chunk, NDVar)
        elif stim_is_single != isinstance(stim_chunk, NDVar):
            raise ValueError(f"stim={stim}: inconsistent element types (NDVar/list)")

        if stim_is_single:
            stim_chunk = [stim_chunk]

        if n_trials:
            if not all(s.has_case and len(s) == n_trials for s in stim_chunk):
                raise ValueError(f"meg={meg}, stim={stim}: inconsistent number of cases")
            stim_trials.extend(zip(*stim_chunk))
        else:
            if any(s.has_case for s in stim_chunk):
                raise ValueError(f"meg={meg}, stim={stim}: inconsistent case dimensions")
            stim_trials.append(stim_chunk)

    # normalize=True defaults to 'l2'
    if normalize is False:
        s_baseline, s_scale = None, None
    elif normalize is True:
        normalize = 'l2'
        s_baseline, s_scale = get_scaling(stim_trials, normalize)
    elif isinstance(normalize, str):
        if normalize not in ('l1', 'l2'):
            raise ValueError(f"normalize={normalize!r}, need bool or 'l1' or 'l2'")
        s_baseline, s_scale = get_scaling(stim_trials, normalize)
    else:
        raise TypeError(f"normalize={normalize!r}, need bool or str")

    # Call `REG_Data.add_data` once for each contiguous segment of MEG data
    ds = RegressionData(tstart, tstop, nlevels, s_baseline, s_scale, stim_is_single, gaussian_fwhm)
    for r, ss in zip(meg_trials, stim_trials):
        if not in_place:
            ss = [s.copy() for s in ss]
        ds.add_data(r, ss)

    # import ipdb; ipdb.set_trace()
    if do_post_normalization:
        ds.post_normalization()
    # import ipdb; ipdb.set_trace()

    # noise covariance

    noise_cov = _handle_noise_channels(noise, ds.sensor_dim)

    # Regularizer Choice
    if isinstance(mu, (tuple, list, np.ndarray)):
        if len(mu) > 1:
            mus = mu
            do_crossvalidation = True
        else:
            mus = None
            do_crossvalidation = False
    elif isinstance(mu, float):
        mus = None
        do_crossvalidation = False
    elif mu == 'auto':
        mus = 'auto'
        do_crossvalidation = True
    else:
        raise ValueError(f"invalid mu={mu!r}, supports tuple, list, np.ndarray or scalar float"
                         f"optionally, may be left 'auto' if not sure!")

    if lead_field.get_dim('sensor') != ds.sensor_dim:
        lead_field = lead_field.sub(sensor=ds.sensor_dim)

    model = NCRF(lead_field, noise_cov, n_iter=n_iter, n_iterc=n_iterc, n_iterf=n_iterf)
    model.fit(ds, mu, do_crossvalidation, tol, verbose, mus=mus, n_splits=n_splits,
              n_workers=n_workers, use_ES=use_ES, compute_explained_variance=True)
    return model


def get_scaling(all_stims: List[List[NDVar]], normalize: str):
    stim_trials = [trials for trials in zip(*all_stims)]  # -> [[stim_1_trial_1, stim_1_trial_2, ...], ...]
    n = sum(len(stim.time) for stim in stim_trials[0])
    means = [sum(s.sum('time') for s in trials) / n for trials in stim_trials]
    stim_trials = [[s - mean for s in stims] for mean, stims in zip(means, stim_trials)]
    if normalize == 'l1':
        scales = [sum(s.abs().sum('time') for s in trials) / n for trials in stim_trials]
    else:
        scales = [(sum((s ** 2).sum('time') for s in trials) / n) ** 0.5 for trials in stim_trials]
    return means, scales
