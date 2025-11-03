"""
Volume Source Example
=====================
Estimate NCRFs for standard and oddball tones.

For this tutorial, we use the auditory Brainstorm tutorial dataset :cite:`Brainstorm` that is available as a part of the Brainstorm software.

.. contents:: Contents
   :local:

.. note::
   Downloading the dataset requires answering an interactive prompt (see
   :func:`mne.datasets.brainstorm.bst_auditory.data_path`).
"""
# Authors: Proloy Das <proloy@umd.edu>
#          Christian Brodbeck <brodbecc@mcmaster.ca>
#
# sphinx_gallery_thumbnail_number = 8

import eelbrain
import mne
from ncrf import fit_ncrf
import numpy as np
import pandas as pd
###############################################################################
# Preprocessing
# -------------
# Preprocess MEG Data: low pass filtering, power line attenuation, downsampling, etc.
# We broadly follow `this mne-python tutorial <https://mne.tools/stable/auto_tutorials/io/60_ctf_bst_auditory.html>`_.

data_path = mne.datasets.brainstorm.bst_auditory.data_path()
raw_fname = data_path / 'MEG' / 'bst_auditory' / 'S01_AEF_20131218_01.ds'
raw = mne.io.read_raw_ctf(raw_fname, preload=False)

# We mark a set of bad channels that seem noisier than others. 
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

###############################################################################
# Event preprocessing: In this dataset, triggers can be adjusted by using a recording of the audio that was presented.

# Events are the presentation times of the audio stimuli: UPPT001
event_fname = data_path / 'MEG' / 'bst_auditory' / 'S01_AEF_20131218_01-eve.fif'
events = mne.find_events(raw, stim_channel='UPPT001')

# The event timing is adjusted by comparing the trigger times on detected sound onsets on channel UADC001-4408.
sound_data = raw[raw.ch_names.index('UADC001-4408')][0][0]
onsets = np.where(np.abs(sound_data) > 2. * np.std(sound_data))[0]
min_diff = int(0.5 * raw.info['sfreq'])
diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
onsets = onsets[diffs > min_diff]
assert len(onsets) == len(events)
diffs = 1000. * (events[:, 0] - onsets) / raw.info['sfreq']
print(f'Trigger delay removed (μ ± σ): {diffs.mean():.1f} ± {diffs.std():.1f} ms')

# Since event onsets are stored in samples
event_sfreq = raw.info['sfreq']

# DataFram for sound events
sound_events = pd.DataFrame({
    'onset': onsets,
    'time': onsets / event_sfreq,
    'duration': np.ones(len(onsets)),
    'id': events[:, 2],
    'label': pd.Categorical.from_codes(events[:, 2], ['', 'standard', 'deviant']),
})
sound_events

###############################################################################
# Preprocess raw data:

# Notch filter for power line artifact
raw.load_data()
raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')

# Band pass filtering 1-8 Hz
raw.filter(1.0, 8.0, fir_design='firwin')

# Crop relative to sound events
tmin = sound_events['time'].min() - 0.2
tmax = sound_events['time'].max() + 1.0
raw.crop(tmin, tmax)

# Resample to 100 Hz
raw.resample(100, npad="auto")

# `raw` remembers the original time axis, consistent with events
raw.first_time

###############################################################################
# Convert MEG data to :class:`eelbrain.NDVar` for NCRF estimation

meg = eelbrain.load.mne.raw_ndvar(raw)
# Time axis is preserved (t_start, t_step, n_samples)
meg.time

###############################################################################
# Continuous stimulus variable construction
# -----------------------------------------
# After loading and preprocessing the raw data, we will construct predictor variables for this particular experiment.
# Here, we construct predictor variables by placing an impulse at every stimulus tone onset. 
# Note that the predictor variable and MEG response should have the same time axis. 
#
# In this example, we use two different predictor variables:
# For the common response to any tone, we put impulses at the presentation times of both the audio stimuli (i.e., standard and deviant beeps).
# To distinguish deviant from standard beeps, we assign 1 and -1 impulses to deviant and standard beeps, respectively (contrast coding).

# Create an all-zero NDVar with time axis matching the MEG data 
stim1 = eelbrain.NDVar.zeros(meg.time, 'common')
# Set values at time points for any sound to 1.  
stim1[sound_events['time'].values] = 1.

# To distinguish deviant from standard beeps, we assign 1 and -1 impulses respectively.
stim2 = stim1.copy('deviant')
standard_index = sound_events['label'] == 'standard'
stim2[sound_events['time'].values[standard_index]] = -1.

# Visualize the predictors:
p = eelbrain.plot.UTS([stim1, stim2], color='black', stem=True, frame='none', w=10, h=2.5, legend=False)

###############################################################################
# Noise covariance estimation
# ---------------------------
# Here we estimate the noise covariance from empty room data.

noise_path = data_path / 'MEG' / 'bst_auditory' / 'S01_Noise_20131218_01.ds'
raw_empty_room = mne.io.read_raw_ctf(noise_path, preload=True)

# Apply the same pre-processing steps to empty room data
raw_empty_room.notch_filter(np.arange(60, 181, 60), fir_design='firwin')

raw_empty_room.filter(1.0, 8.0, fir_design='firwin')

raw_empty_room.resample(100, npad="auto")

# Compute the noise covariance matrix
noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None, method='shrunk', rank=None)

###############################################################################
# Forward model (aka lead-field matrix)
# -------------------------------------
# Create a volume source space based on a 10 mm voxel grid.

# The paths to FreeSurfer reconstructions
subjects_dir = data_path / 'subjects'
subject = 'bst_auditory'

# mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
#                  brain_surfaces='white', orientation='coronal')

# The transformation file obtained by coregistration
trans = data_path / 'MEG' / 'bst_auditory' / 'bst_auditory-trans.fif'

# Here we look at the head only.
# mne.viz.plot_alignment(raw.info, trans, subject=subject, dig=True,
#                        meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
#                        surfaces='head')

# Cache/reload the source space for faster execution
srcfile = subjects_dir / 'bst_auditory' / 'bem' / 'bst_auditory-vol-10-src.fif'
if srcfile.is_file():
    src = mne.read_source_spaces(srcfile)
else:
    surface = subjects_dir / subject / "bem" / "inner_skull.surf"
    src = mne.setup_volume_source_space(subject,
                                        subjects_dir=subjects_dir,
                                        surface=surface,
                                        pos=10.0,
                                        add_interpolator=False)
    mne.write_source_spaces(srcfile, src, overwrite=True)  # needed for smoothing
src

###############################################################################
# Compute the forward solution:
fwdfile = subjects_dir / 'bst_auditory' / 'bem' / 'bst_auditory-vol-10-fwd.fif'
if fwdfile.is_file():
    fwd = mne.read_forward_solution(fwdfile)
else:
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(subject=subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=2)
    mne.write_forward_solution(fwdfile, fwd)

fwd

###############################################################################
# leadfield matrix
lf = eelbrain.load.mne.forward_operator(fwd, src='vol-10',
                                        subjects_dir=subjects_dir,
                                        parc='aparc+aseg')

###############################################################################
# NCRF estimation
# ---------------
# Now that we have all the required data to estimate NCRFs.
#
# .. note::
#    This example uses simplified settings to speed up estimation:
#
#    1) For this example, we use a fixed regularization parameter (``mu``).
#    For a real experiment, the optimal ``mu`` would be determined by
#    cross-validation (set ``mu='auto'``, which is the default).
#    The optimal ``mu`` will then be stored in ``model.mu``
#    (this is how the ``mu`` used here was determined).
#
#    2) The example forces the estimation to stop after fewer iterations than
#    is recommended (``n_iter``). For stable models, we recommend to use the
#    default setting (``n_iter=10``).

# To speed up the example, we cache the NCRF:
ncrf_file = data_path / 'MEG' / 'bst_auditory' / 'oddball_ncrf_vol.pickle'
if ncrf_file.exists():
    model = eelbrain.load.unpickle(ncrf_file)
else:
    model = fit_ncrf(
        meg, [stim1, stim2], lf, noise_cov, tstart=0, tstop=0.5,
        mu=0.000197542704429464, n_iter=5,
    )
    eelbrain.save.pickle(model, ncrf_file)
model

###############################################################################
# The learned kernel/filter (the NCRF) can be accessed as an attribute of the
# ``model``.
# NCRFs are stored as :class:`eelbrain.NDVar`. Here, the two NCRFs correspond
# to the two different predictor variables:

model.h

###############################################################################
# Visualization
# -------------
# A butterfly plot shows weights in all sources over time.
# This is good for forming a quick impression of important time lags,
# or peaks in the response:
#
# .. note::
#    Since the estimates are sparse over cortical locations, smoothing the NCRFs over sources to make the visualization more intuitive.

hs = [h.smooth('source', 0.01, 'gaussian') for h in model.h]
p = eelbrain.plot.Butterfly([h.norm('space') for h in hs], rows=1)

###############################################################################
# The following code for plotting the anatomical localization.
# A single time point can be visualized with the function
# :class:`eelbrain.plot.GlassBrain`:
# First, we locate the sources that are involved in the two prominent early
# peaks in the common response.
times = (0.090, 0.150, 0.200)
vmax = 5e-11
# vmax = hs[0].norm('space').max()  # alternative: vmax based on data
bf = eelbrain.plot.Butterfly(hs[0].norm('space'), h=2, vmax=vmax, ylabel='Amplitude')
for time in times:
    bf.add_vline(time, color='k', linestyle='--')
bs = [eelbrain.plot.GlassBrain(
        hs[0].sub(time=time), vmax=vmax, display_mode='lr', 
        title=f"Common-[{time*1000:.0f} ms]",
      ) for time in times]

###############################################################################
# Next, we do the same with NCRFs to deviant stimulus code.
times = (0.190,)
bf = eelbrain.plot.Butterfly(hs[1].norm('space'), h=2, vmax=vmax, ylabel='Amplitude')
for time in times:
    bf.add_vline(time, color='k', linestyle='--')

bs = [eelbrain.plot.GlassBrain(
        hs[1].sub(time=time), vmax=vmax, display_mode='lr', 
        title=f"Differential-[{time*1000:.0f} ms]",
      ) for time in times]

###############################################################################
# Finally, we can reconstruct the response to standard and deviant stimuli as
# :math:`[Common - Deviant]` amd :math:`[Common + Deviant]` respectively.
times = (0.19,)
# Frequent
h = hs[0] - hs[1]
bf = eelbrain.plot.Butterfly(h.norm('space'), h=2, vmax=vmax, ylabel='Amplitude')
for time in times:
    bf.add_vline(time, color='k', linestyle='--')

bs = [eelbrain.plot.GlassBrain(
        h.sub(time=time), vmax=vmax, display_mode='lr',
        title=f"Frequent-[{time*1000:.0f} ms]",
      ) for time in times]

# Infrequent
h = hs[0] + hs[1]
bf = eelbrain.plot.Butterfly(h.norm('space'), h=2, vmax=vmax, ylabel='Amplitude')
for time in times:
    bf.add_vline(time, color='k', linestyle='--')

bs = [eelbrain.plot.GlassBrain(
        h.sub(time=time), vmax=vmax, display_mode='lr',
        title=f"Infrequent-[{time*1000:.0f} ms]",
      ) for time in times]

###############################################################################
# In an interactive iPython session, we can also use interactive time-linked
# plots with `eelbrain.plot.GlassBrain.butterfly`:

# brain, butterfly = eelbrain.plot.GlassBrain.butterfly(h)
