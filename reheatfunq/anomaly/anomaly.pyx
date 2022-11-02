# Code to represent fault-generated heat flow anomalies.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Malte J. Ziebarth
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

cimport cython
import numpy as np
from cython cimport size_t
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, make_shared

cdef extern from "anomaly.hpp" namespace "reheatfunq" nogil:
    cppclass HeatFlowAnomaly:
        double c_i(double x, double y, double P_H) const
        void batch_c_i_ptr(size_t N, const double* xy, double* c_i,
                           double P_H) const


cdef extern from "anomaly/ls1980.hpp" namespace "reheatfunq" nogil:
    """
    namespace reheatfunq {
    std::shared_ptr<HeatFlowAnomaly>
    convert_LS1980_shared_ptr(std::shared_ptr<LachenbruchSass1980Anomaly> ptr)
    {
        return ptr;
    }
    }
    """
    cppclass LachenbruchSass1980Anomaly(HeatFlowAnomaly):
        LachenbruchSass1980Anomaly(const double* xy, size_t N, double d)

    shared_ptr[HeatFlowAnomaly] \
    convert_LS1980_shared_ptr(shared_ptr[LachenbruchSass1980Anomaly])

ctypedef const double* cdblptr


cdef class Anomaly:
    """
    A heat flow anomaly.
    """
    cdef shared_ptr[HeatFlowAnomaly] _anomaly

    def __call__(self, const double[:,::1] xy, double P_H = 1.0):
        # Sanity:
        if xy.shape[1] != 2:
            raise RuntimeError("`xy` must be of shape (N,2).")
        cdef size_t N = xy.shape[0]
        cdef double[::1] c_i = np.empty(N)
        cdef cdblptr xy_ptr = &xy[0,0] # When used in nogil, emits a warning.
        cdef double* ci_ptr = &c_i[0]

        with nogil:
            deref(self._anomaly).batch_c_i_ptr(N, xy_ptr, ci_ptr, P_H)

        return c_i.base



cdef class AnomalyLS1980(Anomaly):
    """
    A heat flow anomaly
    """

    def __init__(self, const double[:,::1] xy, double d):
        # Sanity:
        if xy.shape[1] != 2:
            raise RuntimeError("`xy` must be of shape (N,2).")
        cdef size_t N = xy.shape[0]
        cdef cdblptr xy_ptr = &xy[0,0]

        with nogil:
            self._anomaly = \
            convert_LS1980_shared_ptr(
                make_shared[LachenbruchSass1980Anomaly](xy_ptr, N, d)
            )