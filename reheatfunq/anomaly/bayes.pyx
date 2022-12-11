# distutils: language = c++
#
# The heat flow anomaly corrected gamma conjugate posterior
# of Ziebarth et al. (2022) [1].
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ,
#               2022 Malte J. Ziebarth
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
from libcpp.vector cimport vector

# The higher precision arithmetics for the posterior-internal
# code are activated via compile-time defines.
# We grab them here for later querying in Python code:
cdef extern from * namespace "reheatfunq::pydefines" nogil:
    """
    namespace reheatfunq {
    namespace pydefines {
    #ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
    bool HAS_FLOAT128 = true;
    #else
    bool HAS_FLOAT128 = false;
    #endif
    #ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
    bool HAS_DEC50 = true;
    #else
    bool HAS_DEC50 = false;
    #endif
    #ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
    bool HAS_DEC100 = true;
    #else
    bool HAS_DEC100 = false;
    #endif
    }
    }
    """
    bool HAS_FLOAT128
    bool HAS_DEC50
    bool HAS_DEC100


cdef extern from "ziebarth2022a.hpp" namespace "pdtoolbox::heatflow" nogil:
    enum precision_t:
        WP_DOUBLE = 0
        WP_LONG_DOUBLE = 1
        WP_FLOAT_128 = 2
        WP_BOOST_DEC_50 = 3
        WP_BOOST_DEC_100 = 4

    void posterior_pdf(const double* x, long double* res, size_t Nx,
                       const double* qi, const double* ci, size_t N, double p,
                       double s, double n, double nu, double amin,
                       double dest_tol, precision_t wp) except+

    void posterior_pdf_batch(const double* x, size_t Nx, long double* res,
                             const vector[const double*]& qi,
                             const vector[const double*]& ci,
                             const vector[size_t]& N,
                             double p, double s, double n, double nu,
                             double amin, double dest_tol,
                             precision_t wp) except+

    void posterior_cdf(const double* x, long double* res, size_t Nx,
                       const double* qi, const double* ci, size_t N, double p,
                       double s, double n, double nu, double amin,
                       double dest_tol, precision_t wp) except+

    void posterior_cdf_batch(const double* x, size_t Nx, long double* res,
                             const vector[const double*]& qi,
                             const vector[const double*]& ci,
                             const vector[size_t]& N,
                             double p, double s, double n, double nu,
                             double amin, double dest_tol,
                             precision_t wp) except+

    void posterior_tail(const double* x, long double* res, size_t Nx,
                        const double* qi, const double* ci, size_t N, double p,
                        double s, double n, double nu, double amin,
                        double dest_tol, precision_t wp) except+

    void posterior_tail_batch(const double* x, size_t Nx, long double* res,
                              const vector[const double*]& qi,
                              const vector[const double*]& ci,
                              const vector[size_t]& N,
                              double p, double s, double n, double nu,
                              double amin, double dest_tol,
                              precision_t wp) except+

    void posterior_log_unnormed(const double* x, long double* res, size_t Nx,
                                const double* qi, const double* ci, size_t N,
                                double p, double s, double n, double nu,
                                double amin, double dest_tol,
                                precision_t wp) except+

    void tail_quantiles(const double* quantiles, double* res,
                        const size_t Nquant, const double* qi, const double* ci,
                        const size_t N, const double p, const double s,
                        const double n, const double nu,
                        const double amin, const double dest_tol) except+

    void posterior_tail_quantiles_batch(
                     const double* quantiles, double* res, const size_t Nquant,
                     const vector[const double*]& qi,
                     const vector[const double*]& ci,
                     const vector[size_t]& N,
                     double p, double s, double n, double nu,
                     double amin, double dest_tol) except+

    int tail_quantiles_intcode(const double* quantiles, double* res,
                               const size_t Nquant, const double* qi,
                               const double* ci, const size_t N, const double p,
                               const double s, const double n, const double nu,
                               const double amin, const double dest_tol,
                               short print) except+


def _support_float128():
    return HAS_FLOAT128

def _support_dec50():
    return HAS_DEC50

def _support_dec100():
    return HAS_DEC100



cdef precision_t get_working_precision(str working_precision, size_t M):
    # Get the working precision.
    if working_precision == 'auto':
        # Decide depending on the number of data points.
        if M > 500:
            working_precision = 'long double'
        else:
            working_precision = 'double'
    cdef precision_t wp
    if working_precision == 'double':
        wp = WP_DOUBLE
    elif working_precision == 'long double':
        wp = WP_LONG_DOUBLE
    elif working_precision == 'float128':
        wp = WP_FLOAT_128
    elif working_precision == 'dec50':
        wp = WP_BOOST_DEC_50
    elif working_precision == 'dec100':
        wp = WP_BOOST_DEC_100
    else:
        raise RuntimeError("Unknonw `working_precision` parameter given.")

    return wp


@cython.boundscheck(False)
def marginal_posterior_pdf(double[::1] P_H, double p, double s, double n,
                           double v, const double[::1] qi, const double[::1] ci,
                           double amin = 1.0, double dest_tol = 1e-8,
                           str working_precision = 'auto'):
    """
    Computes the marginal posterior in total power Q̇ for a given set (s,n,v=ν,p)
    of prior parameters, heat flow measurements qi, and anomaly scalings ci.

    Parameters
    ----------
    ...
    working_precision : str, optional
       Decides in which floating point precision the algorithm should work
       internally. Must be one of 'double', 'long double', 'float128',
       'dec50', 'dec100', or 'auto'. The standard C++ floating point numbers
       are 'double' and 'long double'. The 'dec50' and 'dec100' options refer
       to the boost multiprecision  50 and 100 decimal digit precision numbers,
       respectively. The 'float128' implementation requires  REHEATFUNQ to be
       compiled with the g++ 128bit floating point extension and the define
       'REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD' set.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef long double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")
    z = np.empty(N, dtype=np.longdouble)

    # Working precision:
    cdef precision_t wp = get_working_precision(working_precision, M)

    # Heavy lifting in C++:
    with nogil:
        posterior_pdf(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v, amin,
                      dest_tol, wp)

    return z.base


@cython.boundscheck(False)
def marginal_posterior_pdf_batch(const double[::1] P_H, double p, double s,
                                 double n, double v, list Qi, list Ci,
                                 double amin = 1.0, double dest_tol = 1e-8,
                                 str working_precision = 'auto'):
    """
    ...

    Parameters
    ----------
    ...
    working_precision : str, optional
       Decides in which floating point precision the algorithm should work
       internally. Must be one of 'double', 'long double', 'float128',
       'dec50', 'dec100', or 'auto'. The standard C++ floating point numbers
       are 'double' and 'long double'. The 'dec50' and 'dec100' options refer
       to the boost multiprecision  50 and 100 decimal digit precision numbers,
       respectively. The 'float128' implementation requires  REHEATFUNQ to be
       compiled with the g++ 128bit floating point extension and the define
       'REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD' set.
    """
    cdef size_t Nx = P_H.shape[0]
    cdef size_t Nqc = len(Qi)
    if len(Ci) != Nqc:
        raise RuntimeError("Length of `Qi` and `Ci` needs to match!")

    # Copy references to all the arrays in Qi and Ci:
    cdef const double[::1] qi, ci
    cdef vector[const double*] qi_vec, ci_vec
    cdef vector[size_t] Nqc_i
    cdef Mmax = 0
    qi_vec.resize(Nqc)
    ci_vec.resize(Nqc)
    Nqc_i.resize(Nqc)
    for i in range(Nqc):
        qi = Qi[i]
        ci = Ci[i]
        qi_vec[i] = &qi[0]
        ci_vec[i] = &ci[0]
        Nqc_i[i] = qi.shape[0]
        Mmax = max(qi.shape[0], Mmax)
        if ci.shape[0] != Nqc_i[i]:
            raise RuntimeError("In one sample, the shape of `qi` and `ci` do "
                               "not match.")

    # Working precision:
    cdef precision_t wp = get_working_precision(working_precision, Mmax)

    cdef long double[:,::1] res = np.empty((Nqc, Nx), dtype=np.longdouble)
    with nogil:
        posterior_pdf_batch(&P_H[0], Nx, &res[0,0], qi_vec, ci_vec, Nqc_i,
                            p, s, n, v, amin, dest_tol, wp)

    return res.base



@cython.boundscheck(False)
def marginal_posterior_cdf(double[::1] P_H, double p, double s, double n,
                           double v, const double[::1] qi, const double[::1] ci,
                           double amin = 1.0, double dest_tol=1e-8,
                           str working_precision = 'auto'):
    """
    Computes the marginal posterior cumulative distribution function in
    dissipated power P_H for a given set (s,n,v=ν,p) of prior parameters,
    heat flow measurements qi, and anomaly scalings ci.

    Parameters
    ----------
    ...
    working_precision : str, optional
       Decides in which floating point precision the algorithm should work
       internally. Must be one of 'double', 'long double', 'float128',
       'dec50', 'dec100', or 'auto'. The standard C++ floating point numbers
       are 'double' and 'long double'. The 'dec50' and 'dec100' options refer
       to the boost multiprecision  50 and 100 decimal digit precision numbers,
       respectively. The 'float128' implementation requires  REHEATFUNQ to be
       compiled with the g++ 128bit floating point extension and the define
       'REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD' set.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef long double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")
    z = np.empty(N, dtype=np.longdouble)

    # Working precision:
    cdef precision_t wp = get_working_precision(working_precision, M)

    # Heavy lifting in C++:
    with nogil:
        posterior_cdf(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v,
                      amin, dest_tol, wp)

    return z.base


@cython.boundscheck(False)
def marginal_posterior_cdf_batch(const double[::1] P_H, double p, double s,
                                 double n, double v, list Qi, list Ci,
                                 double amin = 1.0, double dest_tol = 1e-8,
                                 str working_precision = 'auto'):
    """
    ...

    Parameters
    ----------
    ...
    working_precision : str, optional
       Decides in which floating point precision the algorithm should work
       internally. Must be one of 'double', 'long double', 'float128',
       'dec50', 'dec100', or 'auto'. The standard C++ floating point numbers
       are 'double' and 'long double'. The 'dec50' and 'dec100' options refer
       to the boost multiprecision  50 and 100 decimal digit precision numbers,
       respectively. The 'float128' implementation requires  REHEATFUNQ to be
       compiled with the g++ 128bit floating point extension and the define
       'REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD' set.
    """
    cdef size_t Nx = P_H.shape[0]
    cdef size_t Nqc = len(Qi)
    if len(Ci) != Nqc:
        raise RuntimeError("Length of `Qi` and `Ci` needs to match!")

    # Copy references to all the arrays in Qi and Ci:
    cdef const double[::1] qi, ci
    cdef vector[const double*] qi_vec, ci_vec
    cdef vector[size_t] Nqc_i
    cdef size_t Mmax = 0
    qi_vec.resize(Nqc)
    ci_vec.resize(Nqc)
    Nqc_i.resize(Nqc)
    for i in range(Nqc):
        qi = Qi[i]
        ci = Ci[i]
        qi_vec[i] = &qi[0]
        ci_vec[i] = &ci[0]
        Nqc_i[i] = qi.shape[0]
        Mmax = max(qi.shape[0], Mmax)
        if ci.shape[0] != Nqc_i[i]:
            raise RuntimeError("In one sample, the shape of `qi` and `ci` do "
                               "not match.")

    # Working precision:
    cdef precision_t wp = get_working_precision(working_precision, Mmax)

    cdef long double[:,::1] res = np.empty((Nqc, Nx), dtype=np.longdouble)
    with nogil:
        posterior_cdf_batch(&P_H[0], Nx, &res[0,0], qi_vec, ci_vec, Nqc_i,
                            p, s, n, v, amin, dest_tol, wp)

    return res.base


@cython.boundscheck(False)
def marginal_posterior_tail(double[::1] P_H, double p, double s, double n,
                            double v, const double[::1] qi,
                            const double[::1] ci,
                            double amin = 1.0, double dest_tol = 1e-8,
                            str working_precision = 'auto'):
    """
    Computes the marginal posterior cumulative distribution function in
    dissipated power P_H for a given set (s,n,v=ν,p) of prior parameters,
    heat flow measurements qi, and anomaly scalings ci.

    Parameters
    ----------
    ...
    working_precision : str, optional
       Decides in which floating point precision the algorithm should work
       internally. Must be one of 'double', 'long double', 'float128',
       'dec50', 'dec100', or 'auto'. The standard C++ floating point numbers
       are 'double' and 'long double'. The 'dec50' and 'dec100' options refer
       to the boost multiprecision  50 and 100 decimal digit precision numbers,
       respectively. The 'float128' implementation requires  REHEATFUNQ to be
       compiled with the g++ 128bit floating point extension and the define
       'REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD' set.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef long double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")

    z = np.empty(N, dtype=np.longdouble)

    # Working precision:
    cdef precision_t wp = get_working_precision(working_precision, M)

    # Heavy lifting in C++:
    with nogil:
        posterior_tail(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v,
                       amin, dest_tol, wp)

    return z.base


@cython.boundscheck(False)
def marginal_posterior_tail_batch(const double[::1] P_H, double p, double s,
                                  double n, double v, list Qi, list Ci,
                                  double amin = 1.0, double dest_tol = 1e-8,
                                  str working_precision = 'auto'):
    """
    ...

    Parameters
    ----------
    ...
    working_precision : str, optional
       Decides in which floating point precision the algorithm should work
       internally. Must be one of 'double', 'long double', 'float128',
       'dec50', 'dec100', or 'auto'. The standard C++ floating point numbers
       are 'double' and 'long double'. The 'dec50' and 'dec100' options refer
       to the boost multiprecision  50 and 100 decimal digit precision numbers,
       respectively. The 'float128' implementation requires  REHEATFUNQ to be
       compiled with the g++ 128bit floating point extension and the define
       'REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD' set.
    """
    cdef size_t Nx = P_H.shape[0]
    cdef size_t Nqc = len(Qi)
    if len(Ci) != Nqc:
        raise RuntimeError("Length of `Qi` and `Ci` needs to match!")

    # Copy references to all the arrays in Qi and Ci:
    cdef const double[::1] qi, ci
    cdef vector[const double*] qi_vec, ci_vec
    cdef vector[size_t] Nqc_i
    cdef size_t Mmax = 0
    qi_vec.resize(Nqc)
    ci_vec.resize(Nqc)
    Nqc_i.resize(Nqc)
    for i in range(Nqc):
        qi = Qi[i]
        ci = Ci[i]
        qi_vec[i] = &qi[0]
        ci_vec[i] = &ci[0]
        Nqc_i[i] = qi.shape[0]
        Mmax = max(qi.shape[0], Mmax)
        if ci.shape[0] != Nqc_i[i]:
            raise RuntimeError("In one sample, the shape of `qi` and `ci` do "
                               "not match.")

    # Working precision:
    cdef precision_t wp = get_working_precision(working_precision, Mmax)

    cdef long double[:,::1] res = np.empty((Nqc, Nx), dtype=np.longdouble)
    with nogil:
        posterior_tail_batch(&P_H[0], Nx, &res[0,0], qi_vec, ci_vec, Nqc_i,
                             p, s, n, v, amin, dest_tol, wp)

    return res.base



@cython.boundscheck(False)
def marginal_posterior_log_unnormed(double[::1] P_H, double p, double s,
                                    double n, double v, const double[::1] qi,
                                    const double[::1] ci, double epsabs = 1e-8,
                                    double amin = 1.0, double dest_tol=1e-10,
                                    str working_precision = 'auto'):
    """
    Computes the marginal posterior in dissipated power P_H for a given set
    (s,n,v=ν,p) of prior parameters, heat flow measurements qi, and anomaly
    scalings ci.

    Returns the logarithm of the unnormed pdf, i.e. the logarithm of the
    integral over the alpha and beta dimension. This makes it usable for
    adding a new dimension to the posterior and hence normalizing accordingly.

    Parameters
    ----------
    ...
    working_precision : str, optional
       Decides in which floating point precision the algorithm should work
       internally. Must be one of 'double', 'long double', 'float128',
       'dec50', 'dec100', or 'auto'. The standard C++ floating point numbers
       are 'double' and 'long double'. The 'dec50' and 'dec100' options refer
       to the boost multiprecision  50 and 100 decimal digit precision numbers,
       respectively. The 'float128' implementation requires  REHEATFUNQ to be
       compiled with the g++ 128bit floating point extension and the define
       'REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD' set.
    """
    # Step 0: Allocate the memory for evaluating the posterior:
    cdef long double[::1] z
    cdef size_t N = P_H.shape[0], M=qi.shape[0]
    if qi.shape[0] != ci.shape[0]:
        raise RuntimeError("`qi` and `ci` have to be of same shape in "
                           "marginal_posterior.")
    if M == 0:
        raise NotImplementedError("No data given - use prior "
                                  "(not implemented).")

    z = np.empty(N, dtype=np.longdouble)

    # Working precision:
    cdef precision_t wp = get_working_precision(working_precision, M)

    # Heavy lifting in C++:
    with nogil:
        posterior_log_unnormed(&P_H[0], &z[0], N, &qi[0], &ci[0], M, p, s, n, v,
                               amin, dest_tol, wp)

    return z.base


@cython.boundscheck(False)
def marginal_posterior_tail_quantiles(double[::1] quantiles, double p, double s,
                                      double n, double v, const double[::1] qi,
                                      const double[::1] ci,
                                      double amin = 1.0, double dest_tol = 1e-8,
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
                                    M, p, s, n, v, amin, dest_tol,
                                    print_to_cerr)
    if ic != 0:
        raise RuntimeError("Error estimating tail_quantiles.")

    return P_H.base


@cython.boundscheck(False)
def marginal_posterior_tail_quantiles_batch(double[::1] quantiles, double p,
                                            double s, double n, double v,
                                            list Qi, list Ci,
                                            double amin, double dest_tol,
                                            bool inplace = False,
                                            bool print_to_cerr = False):
    """
    Computes the tail quantiles of the marginal posterior in total power P_H
    for a given set (p, s, n, v=ν) of prior parameters, heat flow measurements
    qi, and anomaly scalings ci.

    Returns the total powers P_H[i] at which the complementary CDF has fallen
    to a value quantiles[i].
    """
    cdef size_t Nquant = quantiles.shape[0]
    cdef size_t Nqc = len(Qi)
    if len(Ci) != Nqc:
        raise RuntimeError("Length of `Qi` and `Ci` needs to match!")

    # Copy references to all the arrays in Qi and Ci:
    cdef const double[::1] qi, ci
    cdef vector[const double*] qi_vec, ci_vec
    cdef vector[size_t] Nqc_i
    qi_vec.resize(Nqc)
    ci_vec.resize(Nqc)
    Nqc_i.resize(Nqc)
    for i in range(Nqc):
        qi = Qi[i]
        ci = Ci[i]
        qi_vec[i] = &qi[0]
        ci_vec[i] = &ci[0]
        Nqc_i[i] = qi.shape[0]
        if ci.shape[0] != Nqc_i[i]:
            raise RuntimeError("In one sample, the shape of `qi` and `ci` do "
                               "not match.")

    cdef double[::1] res = quantiles if inplace else np.empty(Nqc)
    with nogil:
        posterior_tail_quantiles_batch(&quantiles[0], &res[0], Nquant, qi_vec,
                                       ci_vec, Nqc_i, p, s, n, v, amin,
                                       dest_tol)

    return res.base