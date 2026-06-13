Contributing
============
Clone the GitHub repository and change the working directory:

``git clone https://github.com/proloyd/neuro-currentRF.git``

``cd neuro-currentRF``

Create an environment with the necessary dependencies (this assumes `Mamba <https://conda-forge.org/download/>`_ is already installed): 

``mamba env create --file=env-dev.yml``

Then, install `neuro-currentRF` in development model using pip:

``pip install -e .``

Using a `-e` installation, changes in `*.py` files will be automatically reflected when you `import ncrf`.
Because Python caches imports, you may need to restart the kernel if you make changes after import. 
Changes in compiled files (`*.pyx`, `*.c`, ...) will not be automatically reflects.
These require re-compilation (by running `pip install -e .` from the repository root folder).