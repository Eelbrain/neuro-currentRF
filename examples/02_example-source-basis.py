"""
Source Basis Example
====================
Estimate NCRFs over smooth, local patches of cortex with a Gaussian basis over
the source space.

This example uses the same data and cortical surface source space as the
:ref:`Surface source Example <sphx_glr_auto_examples_01_example-surf-src.py>`:
responses to frequent and infrequent (oddball) tones from the auditory
Brainstorm tutorial dataset :cite:`Brainstorm`.

.. contents:: Contents
   :local:

.. note::
   Downloading the dataset requires answering an interactive prompt (see
   :func:`mne.datasets.brainstorm.bst_auditory.data_path`).
"""
# Authors: Proloy Das <proloy@umd.edu>
#
# sphinx_gallery_thumbnail_number = 2

import eelbrain
import mne
from ncrf import SourceBasis, fit_ncrf
import numpy as np

###############################################################################
# Data
# ----
# The data are preprocessed as in the
# :ref:`Surface source Example <sphx_glr_auto_examples_01_example-surf-src.py>`,
# which explains the individual steps.

data_path = mne.datasets.brainstorm.bst_auditory.data_path()
raw_fname = data_path / 'MEG' / 'bst_auditory' / 'S01_AEF_20131218_01.ds'
raw = mne.io.read_raw_ctf(raw_fname, preload=False)
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

# Sound onsets, corrected with the audio channel
events = mne.find_events(raw, stim_channel='UPPT001')
sound_data = raw[raw.ch_names.index('UADC001-4408')][0][0]
onsets = np.where(np.abs(sound_data) > 2. * np.std(sound_data))[0]
min_diff = int(0.5 * raw.info['sfreq'])
diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
onsets = onsets[diffs > min_diff]
assert len(onsets) == len(events)
# Onsets in samples at the new sampling frequency, 100 Hz
onsets = np.int64(onsets * 100 / raw.info['sfreq'])
is_frequent = events[:, 2] == 1

# MEG data: 1-8 Hz, resampled to 100 Hz
raw.load_data()
raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
raw.filter(1.0, 8.0, fir_design='firwin')
raw.resample(100, npad="auto")
picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude='bads')
sensor_dim = eelbrain.load.mne.sensor_dim(raw.info, picks=picks)
time = eelbrain.UTS(0, 1 / raw.info['sfreq'], raw.n_times)
meg = eelbrain.NDVar(raw.get_data(picks), (sensor_dim, time))

# Predictors: impulses at all tones (common), and +1/-1 for infrequent/frequent
# tones (contrast)
stim1 = np.zeros(len(time))
stim1[onsets] = 1.
stim2 = stim1.copy()
stim2[onsets[is_frequent]] = -1.
stim1 = eelbrain.NDVar(stim1, time, 'common')
stim2 = eelbrain.NDVar(stim2, time, 'contrast')

# Noise covariance from the empty room recording
noise_path = data_path / 'MEG' / 'bst_auditory' / 'S01_Noise_20131218_01.ds'
raw_empty_room = mne.io.read_raw_ctf(noise_path, preload=True)
raw_empty_room.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
raw_empty_room.filter(1.0, 8.0, fir_design='firwin')
raw_empty_room.resample(100, npad="auto")
noise_cov = mne.compute_raw_covariance(raw_empty_room, method='shrunk', rank=None)

###############################################################################
# Forward model
# -------------
# We use an ``ico-4`` cortical surface source space, with sources oriented
# perpendicular to the cortex. The source basis measures distances along the
# cortex, so the source space needs geodesic distances
# (:func:`mne.add_source_space_distances`), stored in the source space file.

subjects_dir = data_path / 'subjects'
subject = 'bst_auditory'
trans = data_path / 'MEG' / 'bst_auditory' / 'bst_auditory-trans.fif'
srcfile = subjects_dir / subject / 'bem' / f'{subject}-ico-4-src.fif'
fwdfile = subjects_dir / subject / 'bem' / f'{subject}-ico-4-fwd.fif'
if fwdfile.is_file():
    fwd = mne.read_forward_solution(fwdfile)
else:
    src = mne.setup_source_space(subject, spacing='ico4', subjects_dir=subjects_dir, add_dist=False)
    mne.add_source_space_distances(src)
    mne.write_source_spaces(srcfile, src, overwrite=True)
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=2)
    mne.write_forward_solution(fwdfile, fwd)

fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
lf = eelbrain.load.mne.forward_operator(fwd_fixed, src='ico-4', subjects_dir=subjects_dir)

###############################################################################
# The source basis
# ----------------
# Without a source basis, NCRFs are estimated for each source, and the sparsity
# penalty selects few, isolated sources. With a source basis, the NCRFs are
# estimated as sums of Gaussian atoms over the cortex, just like the TRFs are
# estimated as sums of Gaussian atoms over time lags. The sparsity penalty then
# selects few, smooth patches of cortex.
#
# On the cortical surface, each atom is a Gaussian of the geodesic distance
# from its center (the distance along the cortex), so an atom does not jump
# across a sulcus to the opposite bank, nor across the midline to the other
# hemisphere. Here, the standard deviation (``std``) is 5 mm, which is
# similar to the spacing of the ``ico-4`` sources. There is an atom centered on
# every source.
#
# :func:`ncrf.fit_ncrf` creates the basis itself from ``source_basis_std``; here
# we create it ourselves to take a look at it:

source = lf.get_dim('source')
source_basis = SourceBasis.from_source(source, std=0.005)
source_basis

###############################################################################
# The basis is a sparse matrix of shape ``(n_sources, n_atoms)``. Each column
# holds the weights of one atom on the sources. The atom centered in the right
# transverse temporal gyrus (Heschl's gyrus) peaks at its center and decays
# with distance from it:

center = np.flatnonzero(source.parc == 'transversetemporal-rh')[0]
atom = eelbrain.NDVar(source_basis.basis[:, [center]].toarray()[:, 0], source, 'atom')
distance = np.linalg.norm(source.coordinates - source.coordinates[center], axis=1)
index = atom.x > 0
p = eelbrain.plot.Scatter(
    eelbrain.Var(atom.x[index], 'Weight'),
    eelbrain.Var(1000 * distance[index], 'Straight line distance from the center (mm)'), h=3, w=5,
    title="Atom in Heschl's gyrus",
)

###############################################################################
# Since the weights follow the geodesic distance, sources at a given straight
# line distance from the center can carry different weights. To see the atom
# on the cortex (requires a 3D backend):

# brain = eelbrain.plot.brain.brain(atom, surf='inflated', hemi='rh')

###############################################################################
# NCRF estimation
# ---------------
# As in the other examples, we use a fixed regularization parameter (``mu``)
# and fewer iterations than recommended, to speed up estimation. Here, ``mu``
# is one of the candidates that ``mu='auto'`` cross-validates for the model
# with the source basis, the largest one that still leaves a response. Since
# the basis changes the lead field that the solver sees, values of ``mu``
# determined without it do not carry over.

# To speed up the example, we cache the NCRF
ncrf_file = data_path / 'MEG' / 'bst_auditory' / 'oddball_ncrf_source_basis.pickle'
if ncrf_file.exists():
    result = eelbrain.load.unpickle(ncrf_file)
else:
    result = fit_ncrf(
        meg, [stim1, stim2], lf, noise_cov, tstart=0, tstop=0.5,
        mu=0.00017123633556392407, n_iter=5,
        source_basis_std=0.004,
    )
    eelbrain.save.pickle(result, ncrf_file)
result

###############################################################################
# The NCRFs are estimated over the atoms, and :attr:`ncrf.NCRF.h` expands them
# back onto the sources:

result.model.h

###############################################################################
# Visualization
# -------------
# Since the NCRFs are smooth over the cortex, they can be plotted without
# smoothing them first (compare with the
# :ref:`Surface source Example <sphx_glr_auto_examples_01_example-surf-src.py>`):

hs = result.model.h
p = eelbrain.plot.Butterfly(hs, axtitle=['Common', 'Contrast'], rows=1)

###############################################################################
# The response to frequent and infrequent tones are
# :math:`[Common - Contrast]` and :math:`[Common + Contrast]` respectively:

h_recon = [(hs[0] - hs[1]).copy('frequent'), (hs[0] + hs[1]).copy('infrequent')]
p = eelbrain.plot.Butterfly(h_recon, rows=1)

###############################################################################
# Anatomical plots need a 3D backend, which does not work reliably in the
# automatic documentation; uncomment to plot a time point, or a sequence of
# time windows (e.g., in a jupyter notebook):

# brain = eelbrain.plot.brain.brain(hs[0].sub(time=0.095), surf='inflated')
#
# h_binned = hs[0].bin(0.1, 0.1, 0.4, 'extrema')
# sp = eelbrain.plot.brain.SequencePlotter()
# sp.set_brain_args(surf='inflated')
# sp.add_ndvar(h_binned)
# p = sp.plot_table(view='lateral')

###############################################################################
# Choosing the source basis
# -------------------------
# * ``source_basis_std`` sets the spatial scale of the patches. Values around
#   the spacing of the sources enforce local smoothness; much larger values
#   blur distinct sources together. The source space needs geodesic distances
#   up to 5.3 times ``source_basis_std`` (``dist_limit`` in
#   :func:`mne.add_source_space_distances`).
# * For volume source spaces, the atoms are Gaussian in the straight line
#   distance instead, and ``source_basis_stride`` can place atoms on every
#   *n*-th grid point only. On cortical surfaces, there is an atom on every
#   source.
