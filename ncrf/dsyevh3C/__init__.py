"""Bridge to small C/Cython kernels used by the NCRF solver hot paths."""

from .dsyevh3py import compute_gamma_c, eig3
