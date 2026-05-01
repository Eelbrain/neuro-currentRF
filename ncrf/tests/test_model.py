"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
import numpy as np

from ncrf._model import covariate_from_stim
from .fetch import load

from eelbrain import Categorial, concatenate


def test_covariate_from_stim():
    stim = load('stim')[0]
    # Test if difference between list of stimuli and concatenated stimuli
    diff = stim.diff('time')

    start = [-20, -20]
    stop = [20, 20]
    filter_lengths = np.subtract(stop, start) + 1
    covariates = covariate_from_stim([stim, diff], filter_lengths, start)

    conc = concatenate([stim, diff.clip(0)], Categorial('rep', ['on', 'off']))
    covariates_conc = covariate_from_stim(conc, filter_lengths, start)

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

    np.testing.assert_array_equal(covariates[0].T[:, 0:100], covariates_shift[0].T[:, 20:120])  # Beginning
    np.testing.assert_array_equal(covariates[0].T[:, -120:-20], covariates_shift[0].T[:, -100:])  # End
