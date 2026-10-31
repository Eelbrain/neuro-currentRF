API reference
=============

The public ``ncrf`` API provides a convenience function as well as composable
data, estimator, solver, model, and fit objects.

.. currentmodule:: ncrf

High-level interface
--------------------

.. autosummary::
    :toctree: generated/

    fit_ncrf

Fitting pipeline
----------------

.. autosummary::
    :toctree: generated/

    RegressionData
    NCRFEstimator
    fit_model
    CrossValidation
    CVResult
    crossvalidate

Models and fits
---------------

.. autosummary::
    :toctree: generated/

    NCRF
    NCRFFit

Solvers
-------

.. autosummary::
    :toctree: generated/

    Solver
    SolverFit
    ChampLasso
    ChampLassoFit

Metrics
-------

.. autosummary::
    :toctree: generated/

    explained_variance
    l2_error

Model components
----------------

.. autosummary::
    :toctree: generated/

    ForwardModel
    TRFDesign
