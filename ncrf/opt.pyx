"""Cython kernels for the group-sparsity proximal operator used by NCRF."""

# Author: Proloy Das <proloy@umd.edu>
# License: BSD (3-clause)

#cython: boundscheck=False, wraparound=False
# cython: profile=False

cimport cython
from libc.math cimport sqrt
cimport numpy as cnp

ctypedef cnp.int8_t INT8
ctypedef cnp.int64_t INT64
ctypedef cnp.float64_t FLOAT64


cdef Py_ssize_t I = 3
cdef Py_ssize_t J = 3

@cython.cdivision(True)
def cproxg_group(cnp.ndarray[FLOAT64, ndim=3] y,
                 double mu, cnp.ndarray[FLOAT64, ndim=3] out):
    cdef unsigned long i, j 
    cdef double norm, mul

    cdef Py_ssize_t n_voxels = y.shape[0]
    cdef Py_ssize_t n_times = y.shape[2]

    for j in range(n_times):
        for i in range(n_voxels):
            norm = y[i, 0, j] ** 2 + y[i, 1, j] ** 2 + y[i, 2, j] ** 2
            norm = sqrt(norm)

            mul = 1
            if norm > mu:
                mul -= mu / norm
            else:
                mul = 0

            out[i, 0, j] = mul * y[i, 0, j]
            out[i, 1, j] = mul * y[i, 1, j]
            out[i, 2, j] = mul * y[i, 2, j]

    return out
