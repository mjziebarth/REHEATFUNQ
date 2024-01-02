# Code to generate random global R-disk coverings (RGRDC's).
#
# This file is part of the REHEATFUNQ model. It is based on the 'heatflow.pyx'
# file of the ziebarth_et_al_2022_heatflow python module.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
#               2022 Malte J. Ziebarth
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
cimport cython
from cython.operator cimport dereference as deref
from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.functional cimport hash as cpphash, function
from libcpp.memory cimport shared_ptr, make_shared
from libc.math cimport sqrt
from libcpp.algorithm cimport lower_bound
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random.c_distributions cimport random_interval, \
                                          random_standard_uniform
from numpy cimport uint8_t

cdef extern from * namespace "reheatfunq" nogil:
    """
    namespace reheatfunq {

    std::function<size_t(size_t)>
    generate_generator(bitgen_t* bitgen)
    {
        return [bitgen](size_t N) -> size_t
        {
            return random_interval(bitgen, N-1);
        };
    }

    typedef std::function<size_t(size_t)> int_generator_fun_t;

    }
    """
    cdef cppclass int_generator_fun_t:
        pass

    int_generator_fun_t generate_generator(bitgen_t* bitgen)


cdef extern from "coverings/dminpermutations.hpp" namespace "reheatfunq" nogil:
    vector[pair[double, vector[size_t]]] \
        determine_restricted_samples(const double* xy, const size_t N,
                                     const double dmin,
                                     const size_t max_samples,
                                     const size_t max_iter,
                                     shared_ptr[int_generator_fun_t],
                                     cbool extra_debug_checks) except+

    double global_PHmax(const double* xy,
                        const double* PHmax_i,
                        const size_t N,
                        const double dmin) except+


cdef get_generator_stage_1(bitgen):
    if isinstance(bitgen, np.random.Generator):
        bg = bitgen.bit_generator
    elif isinstance(bitgen, np.random.BitGenerator):
        bg = bitgen
    else:
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
cdef void restricted_sample(const double[:,:] xy, double dmin, uint8_t[::1] out,
                            bitgen_t* rng) nogil:
    """
    Computes a random subset of a sample set, removing random nodes of pairs
    that are closer than a limit distance `dmin`.
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
    cdef double dist2, r
    cdef size_t o0, o1
    cdef double xi, yi
    cdef double dmin2 = dmin * dmin
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
            dist2 = (xi-xy[o1,0])**2 + (yi-xy[o1,1])**2
            if dist2 < dmin2:
                out[o1] = True

    # Now invert out (meaning "remove" --> "keep")
    for i in range(N):
        out[i] = not out[i]


@cython.embedsignature(True)
def conforming_data_selection(const double[:,:] xy, double dmin_m, rng=128):
    """
    This methods applies the spatial data filtering technique
    described in the paper, sub-sampling the data so that the
    minimum distance remains above `dmin_m`.

    The selection process for non-conforming data pairs is stochastic
    but reproducible with identical random number generator `rng`.

    Parameters
    ----------
    xy : array_like
        :python:`(N,2)` array of data points in a projected
        Euclidean coordinate system (in m).
    dmin_m : float
        Minimum inter-point distance for the conforming
        selection criterion (in m).
    rng : int | numpy.random.Generator
        A seed or random generator to draw from for
        reproducibility.

    Returns
    -------
    mask : numpy.ndarray
       A mask filtering out non-conforming data points.
    """
    if xy.shape[1] != 2:
        raise RuntimeError("xy has to be shape (N,2).")

    bg = get_generator_stage_1(rng)
    cdef bitgen_t* bitgen = get_generator_stage_2(bg)
    if bitgen == NULL:
        raise RuntimeError("Could not get the generator.")

    cdef uint8_t[::1] mask = np.empty(xy.shape[0], dtype=bool)

    with nogil:
        restricted_sample(xy, dmin_m, mask, bitgen)

    return mask.base


#
# The following is a hack to circumvent Cython limitations on Ubuntu-22.04
# (Github runner)
#
cdef extern from * nogil:
    """
    static void insert_in_m2i(
            std::unordered_map<std::vector<bool>, size_t>& m2i,
            const std::unordered_map<std::vector<bool>,size_t>::iterator& it,
            const std::pair<std::vector<bool>,size_t>& value
    )
    {
        m2i.insert(it, value);
    }
    """
    cdef void insert_in_m2i(unordered_map[vector[cbool], size_t]& m2i,
                            const unordered_map[vector[cbool],
                                                size_t].iterator& it,
                            const pair[vector[cbool],size_t] value)

@cython.boundscheck(False)
@cython.embedsignature(True)
def bootstrap_data_selection(const double[:,::1] xy, double dmin_m, size_t B,
                             rng=127):
    """
    Computes a set of bootstrap samples of heat flow data points
    conforming to the data selection criterion.

    Parameters
    ----------
    xy : array_like
        :python:`(N,2)` array of data points in a projected
        Euclidean coordinate system (in m).
    dmin_m : float
        Minimum inter-point distance for the conforming
        selection criterion (in m).
    B : int
        Number of bootstrap samples to draw.
    rng : int | numpy.random.Generator
        A seed or random generator to draw from for
        reproducibility.

    Returns
    -------
    subselections : list
       A list of index arrays. Each index array lists the indices
       of a conforming data selection within the data array
       :python:`xy`. The number of index arrays is at most
       :python:`B`. Duplicate data selections are returned only
       once.
    """
    # Sanity:
    if xy.shape[1] != 2:
        raise RuntimeError("Shape of xy must be (N,2).")
    cdef size_t N = xy.shape[0]

    # Reproducible random number generation:
    bg = get_generator_stage_1(rng)
    cdef bitgen_t* bitgen = get_generator_stage_2(bg)

    # Compute a conforming subselection:
    cdef list subselections = []
    cdef size_t i,j,n,nsubsel
    cdef uint8_t[::1] mask = np.empty(N, dtype=bool)
    cdef long[::1] newsample
    cdef unordered_map[vector[cbool],size_t] m2i
    cdef unordered_map[vector[cbool],size_t].iterator it
    cdef pair[vector[cbool], size_t] value
    cdef size_t h
    value.first.resize(N)
    nsubsel = 0
    with nogil:
        for i in range(B):
            # 1) Compute a new data set randomly selected from the data
            #    (the bootstrapping part):
            restricted_sample(xy, dmin_m, mask, bitgen)
            n = 0

            for j in range(N):
                if mask[j]:
                    n += 1
                value.first[j] = mask[j]

            # Depending on whether this configuration has already been found,
            # continue:
            it = m2i.find(value.first)
            if it == m2i.end():
                value.second = nsubsel
                insert_in_m2i(m2i, it, value)
                nsubsel += 1
                with gil:
                    newsample = np.empty(n, dtype=np.int64)
                    subselections.append([1.0, newsample.base])
                n = 0
                for j in range(N):
                    if value.first[j]:
                        newsample[n] = j
                        n += 1
            else:
                n = deref(it).second
                with gil:
                    subselections[n][0] += 1.0

    # Now transform the multiplicity to weights:
    for i in range(nsubsel):
        subselections[i][0] /= B

    bitgen = NULL

    return subselections


@cython.boundscheck(False)
@cython.embedsignature(True)
def all_restricted_samples(const double[:,::1] xy, double dmin,
                           size_t max_samples, size_t max_iter,
                           rng=127, cbool extra_debug_checks = False):
    cdef size_t Nxy = xy.shape[0]
    if Nxy == 0:
        raise RuntimeError("Empty coordinates.")

    # Reproducible random number generation:
    cdef bitgen_t* bitgen
    cdef shared_ptr[int_generator_fun_t] gen
    if rng is not None:
        bg = get_generator_stage_1(rng)
        bitgen= get_generator_stage_2(bg)
        gen = make_shared[int_generator_fun_t](generate_generator(bitgen))

    cdef vector[pair[double, vector[size_t]]] rs \
         = determine_restricted_samples(&xy[0,0], xy.shape[0], dmin,
                                        max_samples, max_iter, gen,
                                        extra_debug_checks)

    cdef size_t i,j
    cdef list res
    cdef long[::1] res_i
    res = list()
    for i in range(rs.size()):
        res_i = np.empty(rs[i].second.size(), dtype=np.int64)
        with nogil:
            for j in range(rs[i].second.size()):
                res_i[j] = rs[i].second[j]
        res.append((rs[i].first, res_i.base))

    bitgen = NULL

    return res


@cython.boundscheck(False)
@cython.embedsignature(True)
def determine_global_PHmax(
        const double[:,::1] xy,
        const double[::1] PHmax_i,
        double dmin
    ):
    cdef size_t Nxy = xy.shape[0]
    if Nxy == 0:
        raise RuntimeError("Empty coordinates.")
    if PHmax_i.shape[0] != Nxy:
        raise RuntimeError("Shapes of xy and PHmax_i not compatible.")

    return global_PHmax(&xy[0,0], &PHmax_i[0], Nxy, dmin)



@cython.boundscheck(False)
@cython.embedsignature(True)
def samples_from_discrete_distribution(const double[::1] w, size_t N, rng=127):
    """
    Sample from a discrete probability distribution.
    """
    # Get the RNG:
    cdef bitgen_t* bitgen
    bg = get_generator_stage_1(rng)
    bitgen = get_generator_stage_2(bg)

    # Return array:
    cdef long[::1] res = np.empty(N, dtype=np.int64)

    cdef double W = 0.0, z
    cdef vector[double] cdf
    cdef size_t i, j
    cdef size_t M = w.shape[0]
    with nogil:
        # Compute the CDF:
        cdf.resize(M)
        for i in range(M):
            W += w[i]
            cdf[i] = W
        if cdf.back() < 1.0:
            cdf[M-1] = 1.0
        elif cdf.back() > 1.0:
            for i in range(M):
                cdf[i] /= W

        # Now generate the random numbers:
        for i in range(N):
            z = random_standard_uniform(bitgen)
            j = lower_bound(cdf.begin(), cdf.end(), z) - cdf.begin()
            if j < 0 or j >= M:
                with gil:
                    raise RuntimeError("Computed index out of bounds.")
            res[i] = j

    bitgen = NULL

    return res.base