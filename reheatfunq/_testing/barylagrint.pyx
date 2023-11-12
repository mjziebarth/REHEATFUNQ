# Test the BarycentricLagrangeInterpolator implementation.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2023 Malte J. Ziebarth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from math import inf
cimport cython
from libcpp cimport bool as cbool
from cython.cimports.cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, make_shared, unique_ptr, make_unique
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "utility/pyfunwrap.hpp" namespace "reheatfunq::utility" nogil:
    cdef cppclass PythonFunctionWrapper:
        PythonFunctionWrapper(PyObject* fun) except+
        double operator()(double x) except+

cdef extern from "testing/barylagrint.hpp" namespace "reheatfunq::testing" nogil:
    void test_barycentric_lagrange_interpolator() except+

cdef extern from "numerics/intervall.hpp" nogil:
    """
    typedef reheatfunq::numerics
                      ::PointInInterval<double>
           PII_t;
    """
    cdef cppclass PII_t:
        PII_t(double,double,double)


cdef extern from "numerics/barylagrint.hpp" nogil:
    """
    typedef reheatfunq::numerics
                      ::PiecewiseBarycentricLagrangeInterpolator<double>
            BLI_double_t;
    """
    cdef cppclass BLI_double_t:
        BLI_double_t(PythonFunctionWrapper pfw, double xmin,
                     double xmax, double tol_rel,
                     double tol_abs, double fmin,
                     double fmax, size_t max_splits,
                     unsigned char max_refinements)
        double operator()(PII_t) except+
        vector[vector[pair[double,double]]] get_samples() const

cdef class BarycentricLagrangeInterpolator:
    """
    BarycentricLagrangeInterpolator
    """
    cdef unique_ptr[BLI_double_t] bli
    cdef double xmin
    cdef double xmax

    def __init__(self, fun, double xmin, double xmax, double tol_rel=1e-8,
                 double tol_abs=inf, double fmin=-inf, double fmax=inf,
                 int max_splits=10, int max_refinements=3):
        # Get the function wrapper:
        cdef PyObject* fun_ptr = <PyObject*>fun
        cdef unique_ptr[PythonFunctionWrapper] pfw
        pfw = make_unique[PythonFunctionWrapper](fun_ptr)

        if max_splits < 0:
            raise ValueError("'max_splits' has to be non-negative.")
        if max_refinements < 0:
            raise ValueError("'max_refinements' has to be non-negative.")
        if max_refinements > 255:
            raise ValueError("'max_refinements' out of range (max 255).")

        with nogil:
            self.bli = make_unique[BLI_double_t](deref(pfw), xmin, xmax,
                                                 tol_rel, tol_abs, fmin, fmax,
                                                 max_splits, max_refinements)
            self.xmin = xmin
            self.xmax = xmax

    cdef double _call_double(self, double x) nogil:
        if not self.bli:
            with gil:
                raise RuntimeError("Interpolator not initialized.")
        return deref(self.bli)(PII_t(x, x-self.xmin, self.xmax-x))

    @cython.boundscheck(False)
    cdef void _call_buffer(self, const double[::1] x, double[::1] dest) nogil:
        cdef size_t i
        if not self.bli:
            with gil:
                raise RuntimeError("Interpolator not initialized.")
        if x.shape[0] !=  dest.shape[0]:
            with gil:
                raise RuntimeError("Shapes of input and output buffer do not "
                                   "match in BarycentricLagrangeInterpolator.")
        for i in range(x.shape[0]):
            dest[i] = deref(self.bli)(PII_t(x[i], x[i]-self.xmin, self.xmax-x[i]))

    def __call__(self, x):
        cdef cbool is_float = False
        try:
            x = float(x)
            is_float = True
        except:
            pass
        cdef double[::1] y
        cdef const double[::1] x_
        if is_float:
            return self._call_double(x)
        else:
            x = np.ascontiguousarray(x)
            y = np.empty_like(x)
            x_ = x
            with nogil:
                self._call_buffer(x_, y)
            return y.base

    @cython.boundscheck(False)
    def samples(self):
        """
        Return the samples.
        """
        if not self.bli:
            raise RuntimeError("Interpolator not initialized.")
        cdef vector[vector[pair[double,double]]] S
        S = deref(self.bli).get_samples()
        cdef double[:,::1] res_i
        cdef size_t i,j
        cdef list res = list()
        for i in range(S.size()):
            res_i = np.empty((S[i].size(), 2))
            res.append(res_i.base)
            with nogil:
                for j in range(S[i].size()):
                    res_i[j,0] = S[i][j].first
                    res_i[j,1] = S[i][j].second

        return res




@cython.boundscheck(False)
def barycentric_lagrange_interpolate(const double[::1] x, fun, double xmin,
                                     double xmax, double tol_rel=1e-8,
                                     double tol_abs=inf, double fmin=-inf,
                                     double fmax=inf):
    """
    Use the BarycentricLagrangeInterpolator.
    """
    cdef PyObject* fun_ptr = <PyObject*>fun
    cdef shared_ptr[PythonFunctionWrapper] pfw
    pfw = make_shared[PythonFunctionWrapper](fun_ptr)

    cdef shared_ptr[BLI_double_t] bli
    with nogil:
        bli = make_shared[BLI_double_t](deref(pfw), xmin, xmax, tol_rel,
                                        tol_abs, fmin, fmax)

    # Now interpolate:
    cdef size_t i
    cdef double[::1] res = np.empty(x.shape[0])
    with nogil:
        for i in range(x.shape[0]):
            res[i] = deref(bli)(PII_t(x[i], x[i]-xmin, xmax-x[i]))

    return res.base


def _test_barycentric_lagrange_interpolator():
    test_barycentric_lagrange_interpolator()
