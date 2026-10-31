Guide
=====

The simplest entry point is :func:`ncrf.fit_ncrf`, which accepts Eelbrain
:class:`~eelbrain.NDVar` objects, prepares the regression design, selects a
regularization value when needed, and returns a structured fit report::

    from ncrf import fit_ncrf

    result = fit_ncrf(
        meg,
        stim,
        lead_field,
        noise,
        mu="auto",
        tstop=1.0,
        n_splits=3,
    )
    trf = result.model.h
    training_scores = result.scores

The fitted :class:`~ncrf.NCRF` in ``result.model`` is independent of the
optimizer that produced it. It can predict or evaluate another compatible
:class:`~ncrf.RegressionData` dataset. Optimization diagnostics remain in
``result.solver_fit``, and cross-validation results can be inspected with
:meth:`~ncrf.NCRFFit.cv_info`.


Normalization
-------------

The covariates that an NCRF is fit against can be centered and scaled.
:class:`~ncrf.RegressionData` handles this normalization: the values are recorded
on its ``design`` and applied whenever the covariates are built from the
predictors, and :meth:`~ncrf.RegressionData.from_data` derives them from the data.

Each predictor's mean is subtracted, and each covariate channel is
divided by a scaling factor, determined through the ``scale`` argument:

- ``'spectral'`` (the default): the average spectral norm of the channel's
  covariates. This equalizes covariate scales across predictor variables, so that
  one regularization parameter is appropriate for all of them.
- ``'l1'``/``'l2'``: the predictor's mean absolute deviation or standard
  deviation.
- ``None``: leave the covariates on their raw scale, without centering.

The centering and the ``'l1'``/``'l2'`` factors are properties of the predictor,
and are measured on all of its samples. The ``'spectral'`` norm is a property of
the covariates that were built from it, and is measured on the samples that carry
a complete lag window.

Centering and scaling are a single choice because they only make sense together.
M/EEG data is high-pass filtered, so an uncentered predictor's mean would
contribute a constant ``mean * sum(h)`` to the prediction, which the fit has to
reconcile with zero-mean data -- in effect constraining the response function to
sum to zero. And the scale factors measure variation around the mean rather than
raw magnitude, since a predictor's offset would otherwise dominate its scale and
leave its variation over-regularized.

:attr:`~ncrf.NCRF.h` is in units of the normalized covariates.
:attr:`~ncrf.NCRF.h_scaled` restores the original stimulus scale,
and is the appropriate representation for comparing response
functions across predictors or across models.

Applying a model to new data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fitted model can only be applied to covariates on the scale it was fit on --
multiplying coefficients learned from ``(stim - baseline) / scaling`` with a raw
stimulus would silently produce wrong predictions and scores. New data therefore
has to carry the same normalization as the training data. Apply the model's own
normalization with :meth:`~ncrf.RegressionData.normalize`::

    test_data = RegressionData.from_data(
        [test_meg],
        [[test_stim]],
        tstart=0.0,
        tstop=1.0,
        stim_is_single=True,
        scale=None,
    )
    test_data = test_data.normalize(result.model.design)
    scores = result.model.evaluate(test_data)

Note that this uses the training data's normalization values, which is what
makes the scores comparable; values derived from the test data would put the
covariates on a different scale. The covariates are built from the raw
predictors, so a dataset can be normalized any number of times with any
compatible design; ``scale=None`` merely skips deriving values that
:meth:`~ncrf.RegressionData.normalize` would replace anyway.

:meth:`~ncrf.NCRF.predict`, :meth:`~ncrf.NCRF.evaluate` and
:meth:`~ncrf.NCRF.voxelwise_explained_variance` raise an error when ``data``
carries a different normalization, or when its design describes different
predictors, TRF timings, or a different basis.

Prediction units
^^^^^^^^^^^^^^^^

:meth:`~ncrf.NCRF.predict` returns predictions in the units of the M/EEG data the
dataset was built from, so they can be compared with the original recording
directly.

:meth:`~ncrf.NCRF.evaluate` instead scores in *whitened* sensor space, where the
noise covariance is the identity and channels are therefore comparable (for a
rank-deficient noise covariance this space has only ``rank`` channels, see
:class:`~ncrf.ForwardModel`). This is
the space the solver optimizes in and the one cross-validation compares
candidates in, which is what makes ``result.scores`` and the cross-validation
scores commensurable. Pass ``whitened=True`` to :meth:`~ncrf.NCRF.predict` to see
the predictions the metrics are computed on.
