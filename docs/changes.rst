Changes
=======

.. _changes-0-5:

0.5
---

Bug fixes
^^^^^^^^^

- **Free-orientation fits used a broken convergence check.** A bug in
  the group sparsity penalty pinned one of the inner solver's two convergence
  measures at a constant value, so the inner iterations could only ever stop on
  the other one. TRF estimates from free-orientation source spaces therefore
  change; fixed-orientation fits are unaffected.
- **Source-wise explained variance was wrong for data with more than one
  segment.** :meth:`~ncrf.NCRF.voxelwise_explained_variance` reused one ``theta``
  scratch buffer across segments without resetting it, so from the second segment
  on the baseline residual was computed with the last source's weights still
  zeroed, corrupting every per-source value.
- **``use_ES=True`` depended on the order in which workers returned.** The guard
  for "the cross-fit winner is already the largest ``mu``" compared it with the
  last element of the unordered result list instead of the largest ``mu``, so
  estimation-stability refinement was skipped or applied inconsistently from run
  to run. Selection is now order-independent, and the ``mu`` grid is extended
  before the criterion is applied, so it sees the complete search range.
- **:attr:`ncrf.NCRF.h_scaled` undid the covariate scaling in the wrong
  direction.** The covariates are divided by the scale, so restoring stimulus
  units requires dividing ``h`` by it as well, but ``h_scaled`` multiplied
  instead. Values obtained with ``normalize='l1'`` or ``normalize='l2'`` were
  therefore off by the square of the scale factor.
- **Data whose sensors do not match the lead field is now rejected.** The
  whitening filter and the lead field are indexed by channel position, so data
  with the same channels in a different order was silently attributed to the
  wrong channels. :func:`~ncrf.fit_ncrf` aligned the lead field itself, but the
  component API did not.

Major API changes
^^^^^^^^^^^^^^^^^

Fitting and the fitted model
""""""""""""""""""""""""""""

Fitting is separated into a reusable fitted model, an estimator that owns the
forward model, and a pluggable optimization algorithm (see :doc:`architecture`).

- :func:`~ncrf.fit_ncrf` returns an :class:`~ncrf.NCRFFit` report rather than a
  fitted :class:`~ncrf.NCRF`. The reusable model is :attr:`~ncrf.NCRFFit.model`.
- :class:`~ncrf.NCRF` is now the fitted model: coefficients, forward state and
  design, with :meth:`~ncrf.NCRF.predict`, :meth:`~ncrf.NCRF.evaluate` and
  :meth:`~ncrf.NCRF.voxelwise_explained_variance`. Fitting moved to
  :class:`~ncrf.NCRFEstimator`, which is constructed from the lead field and
  noise covariance. Code doing ``NCRF(lead_field, noise_covariance).fit(data)``
  becomes ``NCRFEstimator.from_lead_field(lead_field, noise_covariance).fit(data, solver)``.
- The optimization algorithm is a :class:`~ncrf.Solver`, configured
  independently of the data and the forward model.
  :class:`~ncrf.ChampLasso` implements the published algorithm and carries
  ``mu``, the iteration counts, ``tol`` and ``use_es``. It also owns the
  ``mu`` search, so ``n_iter``, ``n_iterc``, ``n_iterf``, ``tol``, ``mu`` and
  ``use_ES`` on :func:`~ncrf.fit_ncrf` are ignored when a ``solver`` is passed.
- Cross-validation folds are configured with :class:`~ncrf.CrossValidation`, and
  each candidate's held-out scores are returned as :class:`~ncrf.CVResult`.
- The model metadata that used to live on the fitted object is now a
  :class:`~ncrf.TRFDesign`, and the forward state a :class:`~ncrf.ForwardModel`.

Attributes of the old fitted model map onto the fit report as follows:

============================================  ==============================================
0.4                                           0.5
============================================  ==============================================
``model.h``, ``model.h_scaled``               ``result.model.h``, ``result.model.h_scaled``
``model.theta``                               ``result.model.theta``
``model.mu``                                  ``result.solver.mu``
``model.explained_var``                       ``result.scores['explained_variance']``
``model.residual``                            ``result.scores['cross_fit']``
``model.voxelwise_explained_variance``        ``result.voxelwise_explained_variance``
``model.Gamma``, ``model.Sigma_b``            ``result.solver_fit.gamma``, ``.sigma_b``
``model.err``, ``model.objective_vals``       ``result.solver_fit.history.relative_change``, ``.objective``
``model.cv_info()``                           ``result.cv_info()``
``model.cv_mu()``                             ``result.solver.mu``
``model.tstart``/``tstep``/``tstop``          ``result.model.design.tstart``/``tstep``/``tstop``
``model.basis_std``                           ``result.model.design.basis_std``
``model.stim_baseline``/``stim_scaling``      ``result.model.design.stim_baseline``/``stim_scaling``
============================================  ==============================================

Because the fitted object was restructured, models pickled with earlier versions
can no longer be loaded.

Sensor handling
"""""""""""""""

Channels are now matched strictly by name, and the three channel sets must be
nested: lead field ⊇ noise covariance ⊇ data. The estimator stores the lead
field and noise covariance in full and derives the whitened forward model for
exactly the data's channels when fitting; a data channel missing from the
noise covariance or the lead field, or a noise channel without a lead field,
is an error naming the offending input. Because channel identity is required
for this matching, ``noise`` must be an :class:`mne.Covariance` or an
:class:`eelbrain.NDVar`; a plain :class:`numpy.ndarray` is no longer accepted.

Applying a model to new data
""""""""""""""""""""""""""""

A fitted :class:`~ncrf.NCRF` can be applied to any compatible dataset:
:meth:`~ncrf.NCRF.predict` returns predictions in the units of the M/EEG data,
and :meth:`~ncrf.NCRF.evaluate` scores them in whitened sensor space with one or
more metrics (:func:`~ncrf.explained_variance`, :func:`~ncrf.l2_error`, or a
custom one). The new data has to carry the training data's normalization; apply
:meth:`~ncrf.RegressionData.normalize` with the model's ``design`` (see
:doc:`guide`).

Normalization
"""""""""""""

Normalization is now owned by :class:`~ncrf.RegressionData` and checked when a
fitted model is applied to new data (see :doc:`guide`).

- :func:`~ncrf.fit_ncrf` and :meth:`~ncrf.RegressionData.from_data` take a single
  ``scale`` argument instead of ``normalize`` and
  ``do_post_normalization``/``post_normalize``, which were not independent
  (post-normalization cancelled out ``'l1'``/``'l2'`` scaling):

  ====================  ==========================  ==================================
  ``normalize``         ``do_post_normalization``    now
  ====================  ==========================  ==================================
  ``False`` (default)   ``True`` (default)          ``scale='spectral'`` (default)
  ``'l1'``/``'l2'``     ``True``                    ``scale='spectral'``
  ``'l1'``/``'l2'``     ``False``                   ``scale='l1'``/``'l2'``
  ``False``             ``False``                   ``scale=None``
  ====================  ==========================  ==================================

- Every ``scale`` also subtracts each predictor's mean. Previously the mean was only
  subtracted with ``normalize``, so the default fit scaled the covariates without
  centering the predictors; it now centers them.
- Normalization is applied to the covariates rather than to ``stim``, which is
  never modified.
- The ``in_place`` argument of :func:`~ncrf.fit_ncrf` is removed. ``meg`` and
  ``stim`` are neither modified nor copied: :class:`~ncrf.RegressionData` keeps
  references to the NDVars and derives the arrays the solver consumes from them
  (so they must not be modified in place while it is in use).
- :attr:`ncrf.NCRF.h_scaled` now also undoes spectral scaling; previously ``h``
  from a post-normalized fit was not in stimulus units. (The direction of the
  correction was also wrong; see `Bug fixes`_.)
- Spectral scaling is now also applied to data with a single covariate channel,
  which changes the scale of ``h`` (but not of ``h_scaled``) for such fits. Since
  ``mu`` regularizes coefficients on the covariate scale, values tuned before this
  change are not comparable for those fits.
- The ``'spectral'`` norm is measured on centered covariates, so its value differs
  slightly from the post-normalization factor of earlier versions.
- With ``pad_stim=True``, the zero-padded edge rows now represent a raw stimulus
  of 0: normalization subtracts the same constant offset from every covariate
  row, including rows whose lag window extends beyond the data. Previously the
  stimulus itself was centered before padding, so the padding represented the
  mean stimulus; edge-row covariates, and hence the fitted TRFs, differ for
  such fits.

The TRF basis
"""""""""""""

- ``nlevels``/``nlevel`` is renamed to ``basis_stride`` on :func:`~ncrf.fit_ncrf`
  and :meth:`~ncrf.RegressionData.from_data`. The meaning is
  unchanged: it is the spacing between neighboring Gabor atoms, in samples.
- The value is recorded as :attr:`ncrf.TRFDesign.basis_stride`; it is ``None``
  for a design constructed directly with a custom
  :attr:`~ncrf.TRFDesign.basis`, which no single stride describes.
