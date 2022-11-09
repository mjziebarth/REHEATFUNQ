# Compute the geodesic distance distribution within a sample.
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

cdef extern from "GeographicLib/Geodesic.hpp" namespace "GeographicLib" nogil:
    cdef cppclass Geodesic:
        double Inverse(double lat1, double lon1, double lat2, double lon2,
                       double& s12) const


cdef extern from *:
    """
    #include <GeographicLib/NearestNeighbor.hpp>
    GeographicLib::Geodesic geodesic(GeographicLib::Geodesic::WGS84());
    """
    Geodesic geodesic


cdef double geodistance(double lon0, double lat0, double lon1,
                        double lat1) nogil:
    """
    Computes geodesic distance on the WGS84 ellipsoid using GeographicLib.
    """
    cdef double dist = 0.0
    geodesic.Inverse(lat0, lon0, lat1, lon1, dist)
    return dist



@cython.boundscheck(False)
def distance_distribution(const double[:] lon, const double[:] lat):
    """
    Computes the distance distribution of a sample distribution.
    """
    cdef size_t N = lon.shape[0]
    cdef double[::1] distances = np.empty((N*(N-1)) // 2)
    cdef size_t i,j,k=0
    with nogil:
        for i in range(N):
            for j in range(i+1,N):
                distances[k] = geodistance(lon[i], lat[i], lon[j], lat[j])
                k += 1

    distances.base.sort()
    return distances.base