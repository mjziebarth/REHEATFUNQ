# Cython interface to REHEATFUNQ resilience analysis code.
# This code was originally part of the ziebarth_et_al_2022_heatflow
# Python module developed at GFZ Potsdam, released under the GPL-3.0-or-later.
# You can find this code here:
# https://git.gfz-potsdam.de/ziebarth/ziebarth-et-al-2022-heat-flow-paper-code
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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
from libcpp cimport bool as cbool
from libcpp.vector cimport vector

cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass array_d1 "std::array<double,1>":
        double& operator[](size_t)

    cdef cppclass array_d2 "std::array<double,2>":
        double& operator[](size_t)

    cdef cppclass array_d3 "std::array<double,3>":
        double& operator[](size_t)

    cdef cppclass array_d4 "std::array<double,4>":
        double& operator[](size_t)

    cdef cppclass array_d41 "std::array<double,41>":
        double& operator[](size_t)


cdef extern from "resilience.hpp" namespace "heatflowpaper" nogil:
    cdef struct quantiles_t:
        double proper
        double improper

    cdef vector[quantiles_t] test_performance_1q(size_t N, size_t M,
                                 double P_MW, double K, double T,
                                 double quantile, double PRIOR_P,
                                 double PRIOR_S, double PRIOR_N,
                                 double PRIOR_V, cbool verbose,
                                 cbool show_failures, size_t seed,
                                 unsigned short nthread, double tol) except+

    cdef vector[quantiles_t] test_performance_41q(size_t N, size_t M,
                                 double P_MW, double K, double T,
                                 const array_d41& quantile, double PRIOR_P,
                                 double PRIOR_S, double PRIOR_N,
                                 double PRIOR_V, cbool verbose,
                                 cbool show_failures, size_t seed,
                                 unsigned short nthread, double tol) except+

    cdef vector[quantiles_t] test_performance_mixture_4q(size_t N, size_t M,
                                 double P_MW, double x0, double s0, double a0,
                                 double x1, double s1, double a1,
                                 const array_d4& quantile, double PRIOR_P,
                                 double PRIOR_S, double PRIOR_N,
                                 double PRIOR_V, cbool verbose,
                                 cbool show_failures, size_t seed,
                                 unsigned short nthread, double tol) except+

    cdef vector[quantiles_t] test_performance_mixture_41q(size_t N, size_t M,
                                 double P_MW, double x0, double s0, double a0,
                                 double x1, double s1, double a1,
                                 const array_d41& quantile, double PRIOR_P,
                                 double PRIOR_S, double PRIOR_N,
                                 double PRIOR_V, cbool verbose,
                                 cbool show_failures, size_t seed,
                                 unsigned short nthread, double tol) except+


@cython.boundscheck(False)
def test_performance_cython(long[:] Nset, size_t M, double P_MW, double K,
                            double T, double[:] quantile, double PRIOR_P,
                            double PRIOR_S, double PRIOR_N, double PRIOR_V,
                            short verbose=True, short show_failures=False,
                            size_t seed=848782, short use_cpp_quantiles=True,
                            double tol=1e-3, unsigned char nthread=0):
    """
    Tests the performance of the gamma model (with and without prior) for
    synthetic data sets that do not stem from a gamma distribution.
    """
    cdef size_t i,l,j

    # Other variables:
    cdef size_t Nq = quantile.shape[0]
    cdef size_t N_Nset = Nset.size
    cdef vector[quantiles_t] result

    cdef double[:,:,:,:] res = np.zeros((2, Nset.shape[0], Nq, M))

    cdef array_d41 quant41

    if Nq == 1:
        for i in range(N_Nset):
            if verbose:
                print("i = " + str(i+1) + " / " + str(N_Nset))
            with nogil:
                result = test_performance_1q(Nset[i], M, P_MW, K,
                                             T, quantile[0], PRIOR_P, PRIOR_S,
                                             PRIOR_N, PRIOR_V, verbose,
                                             show_failures, seed, nthread,
                                             tol)
                for l in range(Nq):
                    for j in range(M):
                        res[0,i,l,j] = result[Nq*j+l].proper
                for l in range(Nq):
                    for j in range(M):
                        res[1,i,l,j] = result[Nq*j+l].improper
    elif Nq == 41:
        for i in range(Nq):
            quant41[i] = quantile[i]

        for i in range(N_Nset):
            if verbose:
                print("i = " + str(i+1) + " / " + str(N_Nset))
            with nogil:
                result = test_performance_41q(Nset[i], M, P_MW, K,
                                              T, quant41, PRIOR_P, PRIOR_S,
                                              PRIOR_N, PRIOR_V, verbose,
                                              show_failures, seed, nthread,
                                              tol)
                for l in range(Nq):
                    for j in range(M):
                        res[0,i,l,j] = result[Nq*j+l].proper
                for l in range(Nq):
                    for j in range(M):
                        res[1,i,l,j] = result[Nq*j+l].improper

    return res.base


@cython.boundscheck(False)
def test_performance_mixture_cython(long[:] Nset, size_t M, double P_MW,
                            double x0, double s0, double a0, double x1,
                            double s1, double a1, double[:] quantile,
                            double PRIOR_P, double PRIOR_S, double PRIOR_N,
                            double PRIOR_V,  short verbose=True,
                            short show_failures=False,
                            size_t seed=848782, short use_cpp_quantiles=True,
                            double tol=1e-3):
    """
    Tests the performance of the gamma model (with and without prior) for
    synthetic data sets that do not stem from a gamma distribution.
    """
    cdef size_t i,l,j

    if quantile.size not in (4,41):
        raise RuntimeError("Only size-4 and size-41 quantile arrays "
                           "allowed!")

    # Other variables:
    cdef size_t Nq = quantile.shape[0]
    cdef size_t N_Nset = Nset.size
    cdef array_d4 quant4
    cdef array_d41 quant41
    cdef vector[quantiles_t] result

    if Nq == 4:
        for i in range(Nq):
            quant4[i] = quantile[i]
    elif Nq == 41:
        for i in range(Nq):
            quant41[i] = quantile[i]

    cdef double[:,:,:,:] res = np.zeros((2, Nset.shape[0], Nq, M))

    if Nq == 4:
        for i in range(N_Nset):
            if verbose:
                print("i = " + str(i+1) + " / " + str(N_Nset))
            with nogil:
                result = test_performance_mixture_4q(Nset[i], M, P_MW, x0, s0,
                                 a0, x1, s1, a1, quant4, PRIOR_P, PRIOR_S,
                                 PRIOR_N, PRIOR_V, verbose, show_failures,
                                 seed, 0, tol)
                for l in range(Nq):
                    for j in range(M):
                        res[0,i,l,j] = result[Nq*j+l].proper
                for l in range(Nq):
                    for j in range(M):
                        res[1,i,l,j] = result[Nq*j+l].improper
    elif Nq == 41:
        for i in range(N_Nset):
            if verbose:
                print("i = " + str(i+1) + " / " + str(N_Nset))
            with nogil:
                result = test_performance_mixture_41q(Nset[i], M, P_MW, x0,
                                 s0, a0, x1, s1, a1, quant41, PRIOR_P,
                                 PRIOR_S, PRIOR_N, PRIOR_V, verbose,
                                 show_failures, seed, 0, tol)
                for l in range(Nq):
                    for j in range(M):
                        res[0,i,l,j] = result[Nq*j+l].proper
                for l in range(Nq):
                    for j in range(M):
                        res[1,i,l,j] = result[Nq*j+l].improper
    else:
        raise RuntimeError("Nq = " + str(Nq))

    return res.base