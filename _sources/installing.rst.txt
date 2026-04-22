Installing
==========
NCRF relies on: `Eelbrain` (`Download/ Installation Instructions <https://github.com/christianbrodbeck/Eelbrain/wiki/Installing#release>`_)

We recommend using `mamba` as package manager.

After successfully installing Eelbrain, one can follow either of the methods to install different versions the repository


Installing the latest version
*****************************
The following comment will install the latest version:

``pip install https://github.com/proloyd/neuro-currentRF/archive/refs/heads/master.zip``


Development version from GitHub
*******************************

Clone the GitHub repository and change the working directory:

``git clone https://github.com/proloyd/neuro-currentRF.git``

``cd neuro-currentRF``

Create an environment with the necessary dependencies (this assumes `mamba <https://conda-forge.org/download/>`_) is already installed): 

``mamba env create --file=env-dev.yml``

Then, install neuro-currentRF in development mode using pip:

``pip install -e .``
