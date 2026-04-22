.. NCRF documentation master file, created by
   sphinx-quickstart on Wed Mar 19 12:54:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NCRF documentation
==================

The magnetoencephalography (MEG) response to continuous auditory stimuli, such as speech, is commonly described using a linear filter, the auditory temporal response function (TRF). Though components of the sensor level TRFs have been well characterized, the cortical distributions of the underlying neural responses are not well-understood. In our recent work, we provide a unified framework for determining the TRFs of neural sources directly from the MEG data, by integrating the TRF and distributed forward  source models into one, and casting the joint estimation task as a Bayesian optimization problem. Though the resulting problem emerges as non-convex, we propose efficient solutions that leverage recent advances in evidence maximization. For more details please refer to :cite:`das2020neuro`.

This repository contains the implementation of our direct TRF estimation algorithm in python.

.. toctree::
   :maxdepth: 1

   installing
   changes
   development
   references


.. toctree::
   :maxdepth: 2

   auto_examples/index
   api/index


Neuro-currentRF is maintained by Proloy Das at National Brain Research Centre, Gurgaon and Christian Brodbeck at McMaster University. Current funding: NIH 1R01MH132660-01A1 (2024-); Past funding:  NSF 1552946; NSF 1734892;  DAPRA N6600118240224; NIH R01-DC-014085 (2016-2020).

This repository is free software, covered by the MIT License. However since they have been mainly developed for academic use, the author would appreciate being given academic credit for it. Whenever you use this software to produce a publication or talk, please cite the appropiate references :cite:`das2020neuro`.
