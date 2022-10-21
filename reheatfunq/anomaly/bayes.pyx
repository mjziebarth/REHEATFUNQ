# distutils: language = c++
#
# The heat flow anomaly corrected gamma conjugate posterior
# of Ziebarth et al. (2022) [1].
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ
#
# [1] Ziebarth, M. J., Anderson, J. G., von Specht, S., Heidbach, O., &
#     Cotton, F. (submitted 2021). "Seismic Efficiency and Elastic Power
#     Constrained by Conductive Heat Flow Anomaly: the Case of the San
#     Andreas Fault".
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


# General math routines
cimport cython
import numpy as np
from libcpp cimport bool



@cython.boundscheck(False)
def marginal_posterior_pdf(double[::1] P_H, double p, double s, double n,
                           double v, const double[::1] qi, const double[::1] ci,
                           double dest_tol = 1e-8, bool inplace=False):
    """
    Computes the marginal posterior in total power Q̇ for a given set (s,n,v=ν,p)
    of prior parameters, heat flow measurements qi, and anomaly scalings ci.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")
    if inplace:
        z = P_H
    else:
        z = np.empty(N)

    # Heavy lifting in C++:
    with nogil:
        posterior_pdf(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v,
                      dest_tol)

    return z.base


@cython.boundscheck(False)
def marginal_posterior_cdf(double[::1] P_H, double p, double s, double n,
                           double v, const double[::1] qi, const double[::1] ci,
                           double dest_tol=1e-8, bool inplace=False):
    """
    Computes the marginal posterior cumulative distribution function in
    dissipated power P_H for a given set (s,n,v=ν,p) of prior parameters,
    heat flow measurements qi, and anomaly scalings ci.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")
    if inplace:
        z = P_H
    else:
        z = np.empty(N)

    # Heavy lifting in C++:
    with nogil:
        posterior_cdf(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v,
                      dest_tol)

    return z.base


@cython.boundscheck(False)
def marginal_posterior_tail(double[::1] P_H, double p, double s, double n,
                              double v, const double[::1] qi,
                              const double[::1] ci,
                              double dest_tol = 1e-8, bool inplace=False):
    """
    Computes the marginal posterior cumulative distribution function in
    dissipated power P_H for a given set (s,n,v=ν,p) of prior parameters,
    heat flow measurements qi, and anomaly scalings ci.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")
    if inplace:
        z = P_H
    else:
        z = np.empty(N)

    # Heavy lifting in C++:
    with nogil:
        posterior_tail(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v,
                       dest_tol)

    return z.base



@cython.boundscheck(False)
def marginal_posterior_log_unnormed(double[::1] P_H, double p, double s,
                                    double n, double v, const double[::1] qi,
                                    const double[::1] ci, double epsabs = 1e-8,
                                    double dest_tol=1e-10, bool inplace=False):
    """
    Computes the marginal posterior in dissipated power P_H for a given set
    (s,n,v=ν,p) of prior parameters, heat flow measurements qi, and anomaly
    scalings ci.

    Returns the logarithm of the unnormed pdf, i.e. the logarithm of the
    integral over the alpha and beta dimension. This makes it usable for
    adding a new dimension to the posterior and hence normalizing accordingly.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")
    if inplace:
        z = P_H
    else:
        z = np.empty(N)

    # Heavy lifting in C++:
    with nogil:
        posterior_log_unnormed(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v,
                               dest_tol)

    return z.base


@cython.boundscheck(False)
def marginal_posterior_tail_quantiles(double[::1] quantiles, double p, double s,
                                      double n, double v, const double[::1] qi,
                                      const double[::1] ci,
                                      double dest_tol = 1e-8,
                                      bool inplace = False,
                                      bool print_to_cerr=False):
    """
    Computes the tail quantiles of the marginal posterior in total power P_H
    for a given set (p, s, n, v=ν) of prior parameters, heat flow measurements
    qi, and anomaly scalings ci.

    Returns the total powers P_H[i] at which the complementary CDF has fallen
    to a value quantiles[i].
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef double[::1] P_H
    cdef size_t N = quantiles.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")
    if inplace:
        P_H = quantiles
    else:
        P_H = np.empty(N)

    # Heavy lifting in C++:
    cdef int ic
    with nogil:
        ic = tail_quantiles_intcode(&quantiles[0], &P_H[0], N, &qi[0], &ci[0],
                                   M, p, s, n, v, dest_tol, print_to_cerr)
    if ic != 0:
        raise RuntimeError("Error estimating tail_quantiles.")

    return P_H.base
