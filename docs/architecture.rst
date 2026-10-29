Architecture
============

Neuro-currentRF separates input preparation, forward-model handling,
optimization, and prediction. This keeps the fitted model reusable and allows
optimization algorithms to be changed without changing the prediction API.


Pipeline
--------

The most important classes are:

* :class:`~ncrf.RegressionData` holds the sensor data and predictors together
  with the design, and derives the arrays the solver consumes from them.
* :class:`~ncrf.NCRFEstimator` owns the forward model and whitening transform,
  selects a solver candidate through cross-validation, and runs the final fit.
* :class:`~ncrf.Solver` implementations, such as :class:`~ncrf.ChampLasso`,
  define optimization and candidate-selection behavior.
* :class:`~ncrf.NCRFFit` provides access to the reusable :class:`~ncrf.NCRF` model
  from the selected solver, along with solver-specific fit state, scores, and diagnostics.

The high-level data flow is::

    M/EEG + stimulus + lead field + noise
                    |
                    v
                fit_ncrf
                    |
          +---------+----------+
          |                    |
    RegressionData       NCRFEstimator
    (M/EEG, predictors    (forward model and
     and TRF design)       whitening)
          |                    |
          +---------+----------+
                    |
           Solver.search
        (optional cross-validation)
                    |
             final solver run
                    |
                    v
                 NCRFFit
          +---------+----------+
          |                    |
        NCRF                SolverFit
    (prediction and       (optimizer state
     reconstructed TRFs)   and diagnostics)

:func:`~ncrf.fit_ncrf` is the convenience layer. It accepts the supported input
layouts, creates :class:`~ncrf.RegressionData`, and delegates the fit to
:class:`~ncrf.NCRFEstimator`.

Data and design
---------------

:class:`~ncrf.RegressionData` holds the M/EEG segments and their predictors in
Eelbrain's deconvolution dataset, which is shared with :func:`eelbrain.boosting`:
one array per segment, which is a view into the input NDVar data where possible,
and the normalization of the predictors (recorded on the design as
``stim_baseline`` and ``stim_scaling``). The design further records the TRF lags,
basis, and predictor dimensions and names. The lagged, basis-projected covariates
the solver consumes are derived from the segments and the design on demand, once
per segment; datasets derived from one another (cross-validation folds, whitened
data) share them. The same compact metadata is stored on the fitted model, so
that ``NCRF.h`` can reconstruct labeled source-space TRFs without retaining the
training dataset, and so that the model can check that new data is on the scale
it was fit on.

:class:`~ncrf.NCRFEstimator` holds a lead field and a sensor noise covariance,
matched by channel name: the noise channels have to be a subset of the lead
field's channels. At fit time it derives the whitened forward model for exactly
the sensors of the data, since whitening is not separable per channel; a data
channel missing from the lead field or the noise covariance is an error. The
derived forward model is attached to the whitened :class:`~ncrf.RegressionData`,
and the fitted :class:`~ncrf.NCRF` retains it to apply the same transform when
predicting.

Solvers and cross-validation
----------------------------

A :class:`~ncrf.Solver` is a configuration for estimating the coefficient
matrix (the NCRF). A solver that has more than one configuration to choose from
selects one in :meth:`~ncrf.Solver.search`, using a :class:`~ncrf.CrossValidation`
configuration describing the folds. The data is partitioned with Eelbrain's
cross-validation machinery (shared with :func:`eelbrain.boosting`): the last
few windows of every segment are held out one at a time for validation, and
the candidates are fit to the data preceding the window (``split='ncrf'`` in
:func:`eelbrain.trf.split_data`).
:class:`~ncrf.ChampLasso` uses it to score its ``mu`` grid, extend the
grid when the winner is on a boundary, and apply the estimation-stability
criterion. The estimator then fits the selected configuration on all of the data.

The generic :class:`~ncrf.SolverFit` contains the fitted coefficient matrix;
concrete solvers can add algorithm-specific state and scores. For example,
:class:`~ncrf.ChampLasso` adds its covariance estimates and iteration history.
These details remain in :attr:`ncrf.NCRFFit.solver_fit` rather than becoming
part of the predictive model.

Lower-level fitting
-------------------

Use the component API when data preparation and solver configuration need to be
controlled independently::

    from ncrf import ChampLasso, CrossValidation, NCRFEstimator, RegressionData

    data = RegressionData.from_data(
        meg,
        stim,
        tstart=0.0,
        tstop=1.0,
    )
    estimator = NCRFEstimator.from_lead_field(lead_field, noise_covariance)
    solver = ChampLasso(mu="auto", n_iter=30, n_iterc=10, n_iterf=100)
    result = estimator.fit(
        data,
        solver,
        cv=CrossValidation(n_splits=3),
    )

With a numeric ``mu``, :class:`~ncrf.ChampLasso` exposes a single candidate and
the cross-validation step is skipped.

Model and fit report
--------------------

:class:`~ncrf.NCRFFit` is the report for one fit. Its main attributes have
deliberately separate lifetimes and responsibilities:

``model``
    The fitted :class:`~ncrf.NCRF`, including coefficients, forward state, TRF
    reconstruction, prediction, and solver-independent evaluation.
``solver``
    The fixed solver configuration selected for the final fit.
``solver_fit``
    Solver-specific fitted state and iteration diagnostics.
``scores``
    Training-set prediction metrics plus any scores contributed by the solver.
``voxelwise_explained_variance``
    Optional source-wise training diagnostic.

To evaluate new data, prepare it with the same predictor layout, time step, and
TRF settings, and apply the training-data normalization to it (see
:doc:`guide`)::

    test_data = test_data.normalize(result.model.design)
    predictions = result.model.predict(test_data)
    scores = result.model.evaluate(test_data)

:meth:`ncrf.NCRF.predict` returns predictions in the units of the M/EEG data
(pass ``whitened=True`` for the space the model is fit in), whereas metrics are
computed in whitened sensor space (see :doc:`guide`). Calling
:meth:`ncrf.NCRF.evaluate` with several metrics predicts the dataset only once.
