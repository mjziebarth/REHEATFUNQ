# Code to generate random global R-disk coverings (RGRDC's).
#
# This file is part of the REHEATFUNQ model. It is based on the 'heatflow.pyx'
# file of the ziebarth_et_al_2022_heatflow python module.
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
cimport cython
from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libc.math cimport sqrt
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random.c_distributions cimport random_interval
from numpy cimport uint8_t


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
            dist = sqrt((xi-xy[o1,0])**2 + (yi-xy[o1,1])**2)
            if dist < dmin:
                out[o1] = True

    # Now invert out (meaning "remove" --> "keep")
    for i in range(N):
        out[i] = not out[i]


def conforming_data_selection(const double[:,:] xy, double dmin_m, rng=128):
    """
    This methods applies the spatial data filtering technique
    described in the paper, sub-sampling the data so that the
    minimum distance remains above `dmin_m`.

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

    cdef uint8_t[::1] mask = np.empty(xy.shape[0], dtype=bool)

    with nogil:
        restricted_sample(xy, dmin_m, mask, bitgen)

    return mask.base