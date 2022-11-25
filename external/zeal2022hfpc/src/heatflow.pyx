# Heat flow paper backend code.
#
# This file is part of the ziebarth_et_al_2022_heatflow python module.
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

import numpy as np
from numpy.random cimport bitgen_t
from scipy.spatial import cKDTree
cimport cython
from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libc.math cimport atan, sqrt, M_PI as pi
from libc.stdint cimport uint32_t
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random.c_distributions cimport random_interval
from numpy cimport uint8_t

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

################################################################################
#                                                                              #
#                 Synthetic Heat Flow Data Generation Code                     #
#                                                                              #
################################################################################

cdef extern from "api.hpp" namespace "paperheatflow" nogil:
    
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


@cython.boundscheck(False)
def generate_synthetic_heat_flow_coverings(double[:] k, double[:] t,
                                           long[:] N, long M, double hf_max,
                                           double w0, double x00, double s0,
                                           double x10, double s1, size_t seed,
                                           unsigned short nthread):
    """
    Generate synthetic heat flow coverings.

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



################################################################################
#                                                                              #
#                          Heat Flow Evaluation                                #
#                          ====================                                #
#                                                                              #
#  The code written here is used to evaluate the heat flow anomaly.            #
################################################################################


cdef double C_LS_reference_anomaly_inf(double x_m, double d_m,
                                       double Qbar_x2_W_m) nogil:
    """
    Compute the reference anomaly from Lachenbruch & Sass (1980)
    for a linear increase in source strength with depth.
    It is given in eqn. (A23a)

    Parameters:
       x_m         : Distance from fault trace [m].
       d_m         : Depth of fault [m].
       Qbar_x2_W_m : Paramameter (‾Q * x₂), the average heat production
                     per fault area (‾Q) times the fault depth (x₂).
                     [W/m]

    Returns:
       q : Heat flow evaluated at the x locations [W/m²].
    """
    cdef double Q = 2.0 * Qbar_x2_W_m / d_m
    cdef double q_W_m2 = 0.0
    if x_m == 0.0:
        q_W_m2 = Q / pi
        return q_W_m2

    cdef double z = x_m / d_m
    q_W_m2 = Q / pi * (1.0 - z * atan(1./z))

    return q_W_m2


cdef double C_LS_anomaly_power_scaled(double x_m, double P_W, double L_m,
                                      double d_m) nogil:
    """
    Compute the reference anomaly from Lachenbruch & Sass (1980)
    scaled to a total power release P on the fault.
    """
    # We distribute the power evenly along the
    # fault segment length:
    cdef double Qbar_x2 = P_W / L_m

    # Now everything as in Lachenbruch & Sass (1980):
    cdef double hf_W_m2 = C_LS_reference_anomaly_inf(x_m, d_m, Qbar_x2)

    return hf_W_m2


@cython.boundscheck(False)
def LS_anomaly_power_scaled(const double[:] x_m, double P_W, double L_m,
                            double d_m):
    """
    Compute the reference anomaly from Lachenbruch & Sass (1980)
    scaled to a total power release P on the fault.

    Parameters:
       x_m  : Perpendicular distance from fault [m].
       P_W  : Total power released by the fault [m].
       L_m  : Fault length [m].
       d_m  : Depth of the fault [m].

    Returns:

    """
    cdef size_t N = x_m.size
    cdef size_t i
    cdef double[:] hf_W_m2 = np.empty(N)
    with nogil:
        for i in range(N):
            hf_W_m2[i] = C_LS_anomaly_power_scaled(x_m[i], P_W, L_m, d_m)

    return hf_W_m2.base


@cython.boundscheck(False)
cdef double fault_length(const double[:,:] fault_trace_xy) nogil:
    """
    Computes the fault length.
    """
    cdef size_t i
    cdef size_t M = fault_trace_xy.shape[0]
    cdef double x0,y0,x1,y1
    cdef double L_m = 0.0
    x0 = fault_trace_xy[0,0]
    y0 = fault_trace_xy[0,1]
    for i in range(1,M):
        x1 = fault_trace_xy[i,0]
        y1 = fault_trace_xy[i,1]
        L_m += sqrt((x0-x1)**2 + (y0-y1)**2)
        x0 = x1
        y0 = y1
    return L_m


@cython.boundscheck(False)
cpdef compute_anomaly_LS80(const double[:,:] points_xy,
                           const double[:,:] fault_trace_xy, double P_W,
                           double d_m):
    """
    Computes the reference anomaly from Lachenbruch & Sass (1980)
    scaled to a total power release P on the fault.

    Parameters:
       points_xy      : Array (N,2) of points at which to evaluate the reference
                        anomaly. Must be given in a projected Euclidean
                        coordinate system of unit `m`.
       fault_trace_xy : Array (M,2) (line string) of the fault trace. Must be
                        given in the same coordinate system as points_xy.
       d_m            : Average depth of the fault trace [m].
       P_W            : Total power dissipated on the fault [W].

    Returns:
       hf_W_m2        : Array (N,) of surface heat flow anomaly evaluated at
                        the points xy [W/m²].
    """
    # Sanity:
    if fault_trace_xy.shape[0] == 0:
        raise RuntimeError("Empty fault trace.")
    if points_xy.shape[1] != 2:
        raise RuntimeError("Shape of points_xy must be (N,2).")
    if fault_trace_xy.shape[1] != 2:
        raise RuntimeError("Shape of fault_trace_xy must be (M,2).")
    cdef size_t N = points_xy.shape[0]
    cdef double[:] hf_W_m2 = np.empty(N)

    # Compute the distances from the fault trace.
    cdef object tree = cKDTree(fault_trace_xy)
    cdef double[:] dff = tree.query(points_xy)[0].reshape(-1)
    del tree

    # Compute the fault length:
    cdef size_t M = fault_trace_xy.shape[0]
    cdef double L_m = 0.0
    cdef size_t i
    with nogil:
        L_m = fault_length(fault_trace_xy)


        # Compute the anomaly:
        for i in range(N):
            hf_W_m2[i] = C_LS_anomaly_power_scaled(dff[i], P_W, L_m, d_m)

    return hf_W_m2.base


def compute_ci(const double[:,:] points_xy, const double[:,:] fault_trace_xy,
               double d_m):
    """
    Computes the reference anomaly from Lachenbruch & Sass (1980)
    per unit power release P on the fault.

    Parameters:
       points_xy      : Array (N,2) of points at which to evaluate the reference
                        anomaly. Must be given in a projected Euclidean
                        coordinate system of unit `m`.
       fault_trace_xy : Array (M,2) (line string) of the fault trace. Must be
                        given in the same coordinate system as points_xy.
       d_m            : Average depth of the fault trace [m].

    Returns:
       c_i            : Array (N,) of surface heat flow anomaly per unit power
                        on the fault evaluated at the points xy [1/m²].
    """
    qi = compute_anomaly_LS80(points_xy, fault_trace_xy, 1.0, d_m)
    # c_i = qi / Qdot, but Qdot == 1.0
    # so c_i = q_i (with different unit dimension).
    return qi


################################################################################
#                                                                              #
#                     Data Selection and Bootstrapping                         #
#                     ================================                         #
#                                                                              #
################################################################################

cdef get_generator_stage_1(bitgen):
    if isinstance(bitgen, np.random.Generator):
        print("1")
        bg = bitgen.bit_generator
    elif isinstance(bitgen, np.random.BitGenerator):
        print("2")
        bg = bitgen
    else:
        print("3")
        rng = np.random.default_rng(bitgen)
        bg = rng.bit_generator
    return bg


cdef bitgen_t* get_generator_stage_2(bg) except+:
    """
    Get the bit generator instance. A general sanity check.
    """
    cdef const char *capsule_name = "BitGenerator"
    capsule = bg.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        return NULL

    return <bitgen_t*> PyCapsule_GetPointer(capsule, capsule_name)


@cython.boundscheck(False)
cdef void restricted_sample(const double[:,:] xy, double dmin, uint8_t[:] out,
                            bitgen_t* rng) nogil:
    """
    Computes a random subset of a sample set, removing random nodes of pairs that are
    closer than a limit distance `dmin`.
    """
    # Compute a permutation:
    cdef size_t N = xy.shape[0]
    cdef vector[size_t] order
    order.reserve(N)
    cdef vector[cbool] mask
    mask.resize(N,True)
    cdef size_t i,j,k,l
    for i in range(N):
        # From the N-i remaining free integers, choose one:
        k = random_interval(rng, N-i-1) if i < N-1 else 0

        # Find the k'th free integer:
        l = 0
        j = 1 if mask.at(l) else 0
        while j <= k:
            l += 1
            if mask.at(l):
                j += 1
        mask[l] = False
        order.push_back(l)

    mask.clear()

    # From the start, all points are selected (remove == false):
    for i in range(N):
        out[i] = False

    # Now iteratively add a point to the set of retained points.
    # For each, mark all points within `dmin` as removed.
    cdef double dist, r
    cdef size_t o0, o1
    cdef double xi, yi
    for i in range(N):
        o0 = order.at(i)
        if out[o0]:
            continue
        xi = xy[o0,0]
        yi = xy[o0,1]
        for j in range(i+1,N):
            # Early exit if already removed:
            o1 = order.at(j)
            if out[o1]:
                continue

            # Compute distance:
            dist = sqrt((xi-xy[o1,0])**2 + (yi-xy[o1,1])**2) #geodistance(lon[o0], lat[o0], lon[o1], lat[o1])
            if dist < dmin:
                out[o1] = True

    # Now invert out (meaning "remove" --> "keep")
    for i in range(N):
        out[i] = not out[i]


def conforming_data_selection(const double[:,:] xy, double dmin_m, rng=128):
    """
    This methods applies the spatial data filtering technique
    described in the paper, sub-sampling the data so that the
    minimum distance remains above 20 km.

    The selection process for non-conforming data pairs is stochastic
    but reproducible with identical random number generator `rng`.

    Returns:
       mask : A mask filtering out non-conforming data points.
    """
    if xy.shape[1] != 2:
        raise RuntimeError("xy has to be shape (N,2).")

    bg = get_generator_stage_1(rng)
    cdef bitgen_t* bitgen = get_generator_stage_2(bg)
    if bitgen == NULL:
        raise RuntimeError("Could not get the generator.")

    cdef uint8_t[:] mask = np.empty(xy.shape[0], dtype=bool)

    with nogil:
        restricted_sample(xy, dmin_m, mask, bitgen)

    return mask.base


@cython.boundscheck(False)
def bootstrap_ci(const double[:,:] data_xy, const double[:,:] fault_trace_xy,
                 double d_m, double dmin_m, size_t B, rng=127):
    """
    Computes a set of bootstrap samples of heat flow data points
    conforming to the data selection criterion, and then computes
    the anomaly strengths at the data points.

    Parameters:
       data_xy        : (N,2) array of data points in a projected
                        Euclidean coordinate system.
                        Given in [m].
       fault_trace_xy : (M,2) array of surface trace points.
       d_m            : Fault depth [m]
       L_m            : Fault length [m]
       dmin_m         : Minimum inter-point distance for the
                        conforming selection criterion.
       B              : Number of bootstrap samples.
    """
    # Sanity:
    if fault_trace_xy.shape[0] == 0:
        raise RuntimeError("Empty fault trace.")
    if data_xy.shape[1] != 2:
        raise RuntimeError("Shape of data_xy must be (N,2).")
    if fault_trace_xy.shape[1] != 2:
        raise RuntimeError("Shape of fault_trace_xy must be (M,2).")
    cdef size_t N = data_xy.shape[0]

    # Reproducible random number generation:
    bg = get_generator_stage_1(rng)
    cdef bitgen_t* bitgen = get_generator_stage_2(bg)

    # Compute the anomaly impact c_i on all data points:
    cdef double[:] ci_all \
        = compute_anomaly_LS80(data_xy, fault_trace_xy, 1.0, d_m)

    cdef double L_m = fault_length(fault_trace_xy)

    # Compute a conforming subselection:
    cdef list ci = []
    cdef list masks = []
    cdef size_t i,j,n
    cdef double[:] ci_i
    cdef uint8_t[:] mask
    for i in range(B):
        # 1) Compute a new data set randomly selected from the data
        #    (the bootstrapping part):
        mask = np.empty(N, dtype=bool)
        with nogil:
            restricted_sample(data_xy, dmin_m, mask, bitgen)

        n = 0
        for j in range(N):
            if mask[j]:
                n += 1

        ci_i = np.empty(n)
        n = 0
        for j in range(N):
            if mask[j]:
                ci_i[n] = ci_all[j]
                n += 1

        # 2) Compute the anomaly impact c_i and the advection-correction
        ci.append(ci_i.base)
        masks.append(mask.base)

    bitgen = NULL

    return ci,masks



################################################################################
#                                                                              #
#                            Resilience Analysis                               #
#                            ===================                               #
#                                                                              #
################################################################################

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
        #with nogil:
        for i in range(N_Nset):
            print("i = ",i,"/",N_Nset)
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
            print("i = ",i,"/",N_Nset)
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
            print("i = ",i,"/",N_Nset)
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
            print("i = ",i,"/",N_Nset)
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


