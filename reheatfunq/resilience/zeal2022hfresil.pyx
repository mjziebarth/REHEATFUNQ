# Cython interface to REHEATFUNQ resilience analysis code.
# This code was originally part of the ziebarth_et_al_2022_heatflow
# Python module developed at GFZ Potsdam, released under the GPL-3.0-or-later.
# You can find this code here:
# https://git.gfz-potsdam.de/ziebarth/ziebarth-et-al-2022-heat-flow-paper-code
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
#                    Malte J. Ziebarth
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


################################################################################
#                                                                              #
#             Resilience of the Bayesian analysis to different                 #
#             probability distributions on a sample level.                     #
#                                                                              #
################################################################################


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



################################################################################
#                                                                              #
#         Synthetic Random Global R-Disk Covering Generation Code              #
#                                                                              #
################################################################################

cdef extern from "synthetic_covering.hpp" namespace "paperheatflow" nogil:

    struct gamma_params:
        double k
        double t

    struct sample_params_t:
        size_t N
        gamma_params kt

    vector[vector[vector[double]]] \
    generate_synthetic_heat_flow_coverings_mixture(
        const vector[sample_params_t]&, size_t N, double hf_max, double w0,
        double x00, double s0, double x10, double s1, size_t seed,
        unsigned short nthread
    )

    vector[vector[vector[double]]] \
    generate_synthetic_heat_flow_coverings_mixture(
        const vector[vector[sample_params_t]]&, double hf_max,
        double w0, double x00, double s0, double w1, double x10, double s1,
        double x20, double s2, size_t seed, unsigned short nthread
    )

    vector[double] \
    mixture_normal_3(size_t N, double w0, double x00, double s0, double w1,
                     double x10, double s1, double x20, double s2, size_t seed
    )


@cython.boundscheck(False)
def generate_synthetic_heat_flow_coverings_mix2(
         const double[:] k, const double[:] t, const long[:] N, long M,
         double hf_max, double w0, double x00, double s0, double x10, double s1,
         size_t seed, unsigned short nthread):
    """
    Generate synthetic heat flow coverings using a two component
    normal mixture distribution as an error distribution.

    Parameters:
       k      : Array of gamma distribution parameters `k`.
                shape: (N,)
                dtype: float
       k      : Array of gamma distribution parameters `θ`.
                shape (N,)
                dtype: float
       N      : Array of sample sizes to draw from the corresponding
                gamma distributions.
                shape: (N,)
                dtype: float
       M      : Number of coverings to generate.
                type: int
       hf_max : Threshold below which to accept heat flow values.
                type: float
       w0     : Weight of the first normal distribution describing
                the error mixture distribution.
                type: float
       x00    : Location of the first normal distribution.
                type: float
       s0     : Standard deviation of the first normal distribution.
                type: float
       x10    : Location of the second normal distribution.
                type: float
       s1     : Standard deviation of the second normal distribution.
                type: float
       seed   : Seed by which to initialize the random number generation.
                type:  int
       nthread: Number of threads to use. In combination with seed, this
                fixes the sequence of random number generation used in this
                run. Keep both values the same to obtain reproducible
                results.
                type: int
    """
    cdef size_t n = k.size
    if k.size != t.size:
        raise RuntimeError("k and t have to have the same size.")
    if k.size != N.size:
        raise RuntimeError("k and N have to have the same size.")

    # Copy to C++:
    cdef vector[sample_params_t] params
    cdef size_t i
    cdef vector[vector[vector[double]]] res_cpp
    with nogil:
        params.resize(n)
        for i in range(n):
            params[i].N = N[i]
            params[i].kt.k = k[i]
            params[i].kt.t = t[i]

        res_cpp = generate_synthetic_heat_flow_coverings_mixture(params, M,
                                    hf_max, w0, x00, s0, x10, s1, seed, nthread)

        params.clear()

    # Copy to Python:
    cdef double[:] resij
    cdef list res = [[] for i in range(M)]
    cdef size_t j,l,L
    for i in range(M):
        for j in range(n):
            L = res_cpp[i][j].size()
            resij = np.empty(L)
            with nogil:
                for l in range(L):
                    resij[l] = res_cpp[i][j][l]
            res[i].append(resij.base)

    return res


@cython.boundscheck(False)
def generate_synthetic_heat_flow_coverings_mix3(
         list k, list t, list N,
         double hf_max,
         double w0, double x00, double s0,
         double w1, double x10, double s1,
         double x20, double s2,
         size_t seed, unsigned short nthread):
    """
    Generate synthetic heat flow coverings using a three component
    normal mixture distribution as an error distribution.

    Parameters:
       k      : Array of gamma distribution parameters `k`.
                shape: (N,)
                dtype: float
       k      : Array of gamma distribution parameters `θ`.
                shape (N,)
                dtype: float
       N      : Array of sample sizes to draw from the corresponding
                gamma distributions.
                shape: (N,)
                dtype: float
       M      : Number of coverings to generate.
                type: int
       hf_max : Threshold below which to accept heat flow values.
                type: float
       w0     : Weight of the first normal distribution describing
                the error mixture distribution.
                type: float
       x00    : Location of the first normal distribution.
                type: float
       s0     : Standard deviation of the first normal distribution.
                type: float
       w1     : Weight of the second normal distribution describing
                the error mixture distribution.
                type: float
       x10    : Location of the second normal distribution.
                type: float
       s1     : Standard deviation of the second normal distribution.
                type: float
       x20    : Location of the third normal distribution.
                type: float
       s2     : Standard deviation of the third normal distribution.
                type: float
       seed   : Seed by which to initialize the random number generation.
                type:  int
       nthread: Number of threads to use. In combination with seed, this
                fixes the sequence of random number generation used in this
                run. Keep both values the same to obtain reproducible
                results.
                type: int
    """

    cdef size_t M = len(k)
    if len(t) != M:
        raise RuntimeError("k and t have to have the same size.")
    if len(N) != M:
        raise RuntimeError("k and N have to have the same size.")


    # Copy to C++:
    cdef vector[vector[sample_params_t]] params
    cdef size_t i, j
    cdef vector[vector[vector[double]]] res_cpp

    cdef size_t n
    cdef const double[:] ki
    cdef const double[:] ti
    cdef const long[:] Ni
    params.resize(M)
    for i in range(M):
        ki = k[i]
        ti = t[i]
        Ni = N[i]
        n = ki.shape[0]
        if ti.shape[0] != n:
            raise RuntimeError("k and t have to have the same size.")
        if Ni.shape[0] != n:
            raise RuntimeError("k and N have to have the same size.")

        with nogil:
            params[i].resize(n)
            for j in range(n):
                params[i][j].N = Ni[j]
                params[i][j].kt.k = ki[j]
                params[i][j].kt.t = ti[j]


    with nogil:
        res_cpp = generate_synthetic_heat_flow_coverings_mixture(params, hf_max,
                                    w0, x00, s0, w1, x10, s1, x20, s2, seed,
                                    nthread)

        params.clear()

    # Copy to Python:
    cdef double[:] resij
    cdef list res = [[] for i in range(M)]
    cdef size_t l,L
    for i in range(M):
        n = k[i].shape[0]
        for j in range(n):
            L = res_cpp[i][j].size()
            resij = np.empty(L)
            with nogil:
                for l in range(L):
                    resij[l] = res_cpp[i][j][l]
            res[i].append(resij.base)

    return res

@cython.boundscheck(False)
def generate_normal_mixture_errors_3(size_t N,
         double w0, double x00, double s0,
         double w1, double x10, double s1,
         double x20, double s2, size_t seed):
    """
    Draw random numbers from the three-component normal mixture
    distribution.
    """
    cdef size_t i
    cdef double[::1] X = np.empty(N)
    cdef vector[double] res

    with nogil:
        res = mixture_normal_3(N, w0, x00, s0, w1, x10, s1, x20, s2, seed)
        for i in range(N):
            X[i] = res[i]

    return X.base