Contributing
============
NCRF is hosted on `GitHub <https://github.com/Eelbrain/neuro-currentRF>`_, with development taking place on the ``master`` branch.
Bug reports and feature requests are welcome on the `issue tracker <https://github.com/Eelbrain/neuro-currentRF/issues>`_.

NCRF follows the same conventions as Eelbrain (see the `Eelbrain Contributor's Guide <https://eelbrain.readthedocs.io/en/latest/development.html>`_). The sections below cover what is specific to NCRF.


Development version
*******************
Clone the repository (or your `fork <https://help.github.com/articles/fork-a-repo>`_) and change into it:

.. code-block:: console

    $ git clone https://github.com/Eelbrain/neuro-currentRF.git
    $ cd neuro-currentRF

Then create the ``ncrf`` environment with all dependencies (this assumes `Mamba <https://conda-forge.org/download/>`_ is already installed), and install NCRF in development mode:

.. code-block:: console

    $ mamba env create --file=env-dev.yml
    $ mamba activate ncrf
    $ pip install -e .

With an ``-e`` installation, changes in ``*.py`` files are automatically reflected when you ``import ncrf``.
Because Python caches imports, you may need to restart the kernel if you make changes after import.
Changes in compiled files (``*.pyx``, ``*.c``, ...) are not automatically reflected; they require re-compilation by running ``pip install -e .`` again from the repository root.


Testing
*******
Tests are in ``ncrf/tests``, and are run with `pytest <https://docs.pytest.org/en/stable/how-to/usage.html>`_ from the repository root:

.. code-block:: console

    $ pytest ncrf                                   # run all tests
    $ pytest ncrf -m "not slow"                     # skip the slow test
    $ pytest ncrf/tests/test_model.py                  # run all tests in a specific file
    $ pytest ncrf/tests/test_model.py::test_fit_model  # run a single test

``test_ncrf`` is marked ``slow`` because it downloads the NCRF testing dataset and fits a full model, which takes on the order of 10 minutes (the rest of the suite runs in seconds).
Use ``-m "not slow"`` to deselect it while iterating; the full suite is run by Continuous Integration (CI) on every pull request.
