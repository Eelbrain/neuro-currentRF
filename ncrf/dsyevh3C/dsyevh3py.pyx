# cython: boundscheck=False, wraparound=False
# cython: profile=False
# distutils: language = c++
# Author: Proloy Das <proloy@umd.edu>
"""Cython bridge for 3x3 eigendecomposition and Gamma updates in NCRF.

This module keeps the small dense linear-algebra kernels used inside the
Champagne updates close to compiled code so the Python solver can stay focused
on orchestration.

Contains code from 'Efficient numerical diagonalization of hermitian 3x3 matrices'
governed by following license ( GNU LESSER GENERAL PUBLIC LICENSE Version 2.1)
Copyright (C) 2008 Joachim Kopp
All rights reserved.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
USA
"""

cimport cython
cimport numpy as cnp
from libc.math cimport sqrt

ctypedef cnp.float64_t FLOAT64
cdef Py_ssize_t I = 3
cdef Py_ssize_t J = 3

cdef float r_TOL = 2e-16


cdef extern from "dsyevh3.c":
    int dsyevh3(double A[3][3], double Q[3][3], double w[3])

cdef extern from "dsytrd3.c":
    dsytrd3(double A[3][3], double Q[3][3], double d[3], double e[2])

cdef extern from "dsyevq3.c":
    int dsyevq3(double A[3][3], double Q[3][3], double w[3])

cdef extern from "dsyevc3.c":
    int dsyevc3(double A[3][3], double w[3])


def eig3(FLOAT64[:,:] A, FLOAT64[:,:] Q, FLOAT64[:] w):
    cdef double Ac[3][3]
    cdef double [:, :] Ac_view = Ac
    cdef double Qc[3][3]
    cdef double [:, :] Qc_view = Qc
    cdef double wc[3]
    cdef double [:] wc_view = wc
    cdef int out

    cdef long int i, j

    for i in range(3):
        for j in range(3):
            Ac_view[i, j] = A[i, j]

    out = dsyevh3(Ac, Qc, wc)

    for i in range(3):
        w[i] = wc_view[i]
        for j in range(J):
            Q[i, j] = Qc_view[i, j]

    return out


cdef int mm(double a[3][3], double b[3][3],
            double c[3][3]):
    cdef int i, j, k
    cdef double sum

    for i in range(3):
        for j in range(3):
            sum = 0
            for k in range(3):
                sum += a[i][k] * b[k][j]
            c[i][j] = sum

    return 0


cdef int mmt(double a[3][3], double b[3][3],
             double c[3][3]):
    cdef int i, j, k
    cdef double sum

    for i in range(3):
        for j in range(3):
            sum = 0
            for k in range(3):
                sum += a[i][k] * b[j][k]
            c[i][j] = sum

    return 0


cdef int mtm(double a[3][3], double b[3][3],
            double c[3][3]):
    cdef int i, j, k
    cdef double sum

    for i in range(3):
        for j in range(3):
            sum = 0
            for k in range(3):
                sum += a[k][i] * b[k][j]
            c[i][j] = sum

    return 0


@cython.cdivision(True)
def compute_gamma_c(FLOAT64[:, :] zpy, FLOAT64[:, :] xpy, FLOAT64[:, :] gamma):
    """Computes Gamma_i

    Computes Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
                   = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
                   = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
                   = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    Parameters
    ----------
    z: ndarray
        array of shape (dc, dc)
        auxiliary variable,  z_i

    x: ndarray
        array of shape (dc, dc)
        auxiliary variable, x_i

    Returns
    -------
    ndarray
    array of shape (dc, dc)

    """
    cdef int i, j

    # Data copy to memory view
    cdef double z[3][3]
    cdef double [:, :] zc = z
    cdef double x[3][3]
    cdef double [:, :] xc = x

    for i in range(3):
        for j in range(3):
            xc[i, j] = xpy[i, j]
            zc[i, j] = zpy[i, j]

    cdef double e[3]
    cdef double v[3][3]
    cdef double d[3]
    cdef double u[3][3]
    cdef double temp[3][3]

    cdef double max_elem, TOL_e, TOL_d

    dsyevh3(z, v, e)
    max_elem = max(e, 3)
    TOL_e = r_TOL * max_elem

    for i in range(3):
        if e[i] < TOL_e:
            e[i] = 0
        else:
            e[i] = sqrt(e[i])

    mm(x, v, temp)
    mtm(v, temp, x)

    for i in range(3):
        for j in range(3):
            temp[i][j] = e[i] * x[i][j] * e[j]

    dsyevh3(temp, u, d)
    max_elem = max(d, 3)
    TOL_d = r_TOL * max_elem

    for i in range(3):
        if d[i] < TOL_d:
            d[i] = 0
        else:
            d[i] = sqrt(d[i])

    for j in range(3):
        for i in range(3):
            if e[j] <= 0:
                v[i][j] = 0
            else:
                v[i][j] = v[i][j] / e[j]

    mm(v, u, temp)

    for j in range(3):
        for i in range(3):
            v[i][j] = temp[i][j] * d[j]

    mmt(v, temp, x)

    for i in range(3):
        for j in range(3):
            gamma[i, j] = x[i][j]

    return


cdef max(double* x, int n):
    cdef int i
    cdef double max_elem = x[0]
    for i in range(1, n):
        if x[i] > max_elem:
            max_elem = x[i]
    return max_elem
