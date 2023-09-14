#
# Code to generate "point-of-interest samplings".
#
# This file is part of the REHEATFUNQ model. It is based on the 'heatflow.pyx'
# file of the ziebarth_et_al_2022_heatflow python module.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
#               2022-2023 Malte J. Ziebarth
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
from numpy cimport uint8_t
from numpy.random cimport bitgen_t
from libc.math cimport sin, cos, sqrt
from libcpp.vector cimport vector
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random.c_distributions cimport random_interval, \
                                          random_standard_uniform

#
# Boilerplate to get the NumPy bit generator.
# from rdisks.pyx
#
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

# end boilerplate





#
# The point-of-interest sampling.
#
cdef extern from * nogil:
    """
    struct xy_t {
        double x;
        double y;
    };
    """
    cdef struct xy_t:
        double x
        double y

@cython.boundscheck(False)
def generate_point_of_interest_sampling(size_t N, double p_poi,
                                        double p_follow_up, double L,
                                        double R, rng=128):

    # Get the generator:
    bg = get_generator_stage_1(rng)
    cdef bitgen_t* bitgen = get_generator_stage_2(bg)
    cdef vector[xy_t] xy
    cdef vector[xy_t] xy_poi
    cdef vector[xy_t] xy_follow_up
    cdef xy_t xy_i
    cdef double ri, r, alpha
    cdef double pi = np.pi

    cdef size_t i,j,i0
    cdef double[:,::1] xy_res = np.empty((N,2))
    cdef uint8_t[::1] code = np.empty(N, dtype=np.uint8)
    with nogil:
        xy_i.x = (L-1) * random_standard_uniform(bitgen)
        xy_i.y = (L-1) * random_standard_uniform(bitgen)
        xy_poi.push_back(xy_i)

        i = 1
        while i < N:
            ri = random_standard_uniform(bitgen)
            if ri < p_follow_up:
                # Random point in a circle of radius R around
                # data point xy_poi[j]:
                j = random_interval(bitgen, xy_poi.size()-1)
#                if j == xy_poi.size(): ## TODO REMOVE ME!
#                    with gil:
#                        raise RuntimeError("random_interval includes the max "
#                                           "integer!")
                r = R * sqrt(random_standard_uniform(bitgen))
                alpha = 2 * pi * random_standard_uniform(bitgen)
                xy_i.x = r * cos(alpha) + xy_poi[j].x
                xy_i.y = r * sin(alpha) + xy_poi[j].y

                # Make sure the point is in bounds:
                if xy_i.x < 0.0 or xy_i.x > L-1:
                    continue
                if xy_i.y < 0.0 or xy_i.y > L-1:
                    continue

                # If in bounds, append:
                xy_follow_up.push_back(xy_i)

            elif ri < p_follow_up + p_poi:
                # Point of interest:
                xy_i.x = (L-1) * random_standard_uniform(bitgen)
                xy_i.y = (L-1) * random_standard_uniform(bitgen)
                xy_poi.push_back(xy_i)
            else:
                # Just a normal sampling point.
                xy_i.x = (L-1) * random_standard_uniform(bitgen)
                xy_i.y = (L-1) * random_standard_uniform(bitgen)
                xy.push_back(xy_i)
            i += 1

        # Transfer:
        i0 = 0
        for i in range(xy.size()):
            xy_res[i,0] = xy[i].x
            xy_res[i,1] = xy[i].y
            code[i] = 0
        i0 += xy.size()
        for i in range(xy_poi.size()):
            xy_res[i+i0, 0] = xy_poi[i].x
            xy_res[i+i0, 1] = xy_poi[i].y
            code[i+i0] = 1
        i0 += xy_poi.size()
        for i in range(xy_follow_up.size()):
            xy_res[i+i0, 0] = xy_follow_up[i].x
            xy_res[i+i0, 1] = xy_follow_up[i].y
            code[i+i0] = 2

    return xy_res.base, code.base