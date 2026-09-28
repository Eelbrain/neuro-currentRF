Architecture
============

Neuro-currentRF separates input preparation, forward-model handling,
optimization, and prediction. This keeps the fitted model reusable and allows
optimization algorithms to be changed without changing the prediction API.


Pipeline
--------

The most important classes are:

* :class:`~ncrf.RegressionData` prepares and stores the sensor data, lagged
  basis-projected covariates, and the metadata needed to reconstruct TRFs.
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
    (TRF design and       (forward model and
     covariates)           whitening)
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

:class:`~ncrf.RegressionData` contains numeric sensor arrays and lagged stimulus
covariates projected into a Gaussian basis. Its design metadata records the TRF
lags, basis, predictor dimensions and names, and the normalization the covariates
carry (``stim_baseline`` and ``stim_scaling``, applied by
:meth:`~ncrf.RegressionData.normalize`). The same compact metadata is stored on
the fitted model, so that ``NCRF.h`` can reconstruct labeled source-space TRFs
without retaining the training dataset, and so that the model can check that new
data is on the scale it was fit on.

:class:`~ncrf.NCRFEstimator` holds a lead field and a sensor noise covariance,
matched by channel name: the noise channels have to be a subset of the lead
field's channels. At fit time it derives the whitened forward model for exactly
the sensors of the data, since whitening is not separable per channel; a data
channel missing from the lead field or the noise covariance is an error. The
derived forward model is not stored on the estimator. The estimator whitens the
data before candidate selection and fitting; the fitted :class:`~ncrf.NCRF`
retains the forward state needed to apply the same transform when predicting.

Solvers and cross-validation
----------------------------

A :class:`~ncrf.Solver` is a configuration for estimating the coefficient
matrix (the NCRF). A solver that has more than one configuration to choose from
selects one in :meth:`~ncrf.Solver.search`, using a :class:`~ncrf.CrossValidation`
configuration describing the folds.
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
        [meg],
        [[stim]],
        tstart=0.0,
        tstop=1.0,
        stim_is_single=True,
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
