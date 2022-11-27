# Maximum likelihood estimators, log-likelihoods, Kullback-Leibler divergence,
# PDFs & CDFs for the gamma conjugate prior and some for the gamma distribution.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2020-2022 GeoForschungsZentrum GFZ,
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

# Python imports:
import numpy as np
from scipy.optimize import minimize
from warnings import warn


# Cython imports:
cimport numpy as cnp
cimport cython
from scipy.special import gamma
from libc.math cimport log, sqrt, fabs, exp, pow, ceil, log1p, isnan, lgamma
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr, shared_ptr, make_shared
from cython.operator cimport dereference as deref
from cython.parallel cimport prange

cdef extern from "gamma_conjugate_prior.hpp" namespace "pdtoolbox" nogil:
    """
    namespace pdtoolbox {
    double _gcp_ln_Phi(double lp, double ls, double n, double v,
                       double amin, double epsrel=1e-10)
    {
        return GammaConjugatePriorBase::ln_Phi(lp, ls, n, v, amin, 0.0, epsrel);
    }

    double _log_gamma_function(double x)
    {
        return std::lgamma(x);
    }

    }
    """
    cdef cppclass GammaConjugatePriorBase:
        GammaConjugatePriorBase(double lp, double s, double n, double v,
                                double amin = 1.0, double epsabs=0,
                                double epsrel=1e-10);

        @staticmethod
        double ln_Phi(double lp, double ls, double n, double v, double amin,
                      double epsabs, double epsrel) except+

        @staticmethod
        double kullback_leibler(double lp, double s, double n, double v,
                                double lp_ref, double ls_ref, double n_ref,
                                double v_ref, double amin, double epsabs,
                                double epsrel) except+

        @staticmethod
        void posterior_predictive_pdf(size_t Nq, const double* q, double* out,
                                      double lp, double s, double n, double v,
                                      double amin, double epsabs,
                                      double epsrel) except+

        @staticmethod
        void posterior_predictive_pdf_batch(size_t Nq, const double* q,
                                double* out, size_t Mparam, const double* lp,
                                const double* s, const double* n,
                                const double* v, double amin, double epsabs,
                                double epsrel) except+

        @staticmethod
        void posterior_predictive_cdf(size_t Nq, const double* q, double* out,
                                      double lp, double s, double n, double v,
                                      double amin, double epsabs,
                                      double epsrel) except+

        @staticmethod
        void posterior_predictive_cdf_batch(size_t Nq, const double* q,
                                double* out, size_t Mparam, const double* lp,
                                const double* s, const double* n,
                                const double* v, double amin, double epsabs,
                                double epsrel) except+

    cdef double _gcp_ln_Phi(double lp, double ls, double n, double v,
                            double amin, double epsrel) except+

    cdef double _gcp_ln_Phi(double lp, double ls, double n, double v,
                            double amin) except+

    cdef double _log_gamma_function(double x) except+



cdef extern from "ll_gamma_conjugate_prior.hpp" namespace "pdtoolbox" nogil:
    cdef cppclass GammaConjugatePriorLogLikelihood:
        @staticmethod
        unique_ptr[GammaConjugatePriorLogLikelihood] \
           make_unique(double p, double s, double n, double v, const double* a,
                       const double* b, size_t Nab, double nv_surplus_min,
                       double vmin, double amin, double epsabs,
                       double epsrel) except+

        bool optimize() except+

        double lp() const
        double s() const
        double n() const
        double v() const


cdef extern from "gamma.hpp" namespace "pdtoolbox" nogil:
    cdef struct gamma_mle_t:
        double a
        double b

    cdef gamma_mle_t compute_gamma_mle(double mean_x, double mean_logx,
                                       double amin)




### Gamma conjugate prior ###

@cython.boundscheck(False)
def gamma_conjugate_prior_mle(const double[::1] a, const double[::1] b,
                              double p0 = 1.0, double s0 = 1.0, double n0 = 1.5,
                              double v0 = 1.0, double nv_surplus_min = 1e-3,
                              double vmin = 0.1, double amin = 1.0,
                              double epsabs = 0.0, double epsrel = 1e-10):
    """
    Maximum likelihood estimator of the gamma conjugate prior.

    Returns:
      p, s, n, v
    """
    # Sanity:
    cdef size_t N = a.shape[0]
    if b.shape[0] != N:
        raise RuntimeError("`a` and `b` need to be of same shape.")

    # The log-likelihood:
    cdef shared_ptr[GammaConjugatePriorLogLikelihood] ll
    cdef const double* w = NULL
    with nogil:
        ll = make_shared[GammaConjugatePriorLogLikelihood](p0, s0, n0, v0,
                                                           &a[0], &b[0], N,
                                                           nv_surplus_min, vmin,
                                                           amin, epsabs, epsrel)
        if ll:
            deref(ll).optimize()
    if not ll:
        raise RuntimeError("Creating the gamma conjugate prior likelihood "
                           "failed.")

    return deref(ll).lp(), deref(ll).s(), deref(ll).n(), deref(ll).v()


@cython.boundscheck(False)
def gamma_conjugate_prior_logL(double[::1] a, double[::1] b, double lp,
                               double s, double n, double v, double amin = 1.0):
    """
    Log-likelihood of the gamma conjugate prior.

    Returns:
       ll : Log-Likelihood (double)
    """
    # Sanity:
    cdef size_t N = a.shape[0]
    if N != b.shape[0]:
        raise RuntimeError("Shapes of `a` and `b` do not match.")

    cdef size_t i
    cdef double ll
    with nogil:
        # Compute the normalization:
        ll = -_gcp_ln_Phi(lp, log(s), n, v, amin)

        # Compute the log-likelihood:
        for i in range(N):
            ll += ((v*a[i] - 1.0) * log(b[i]) + (a[i] - 1.0) * lp - s*b[i]
                   - n * lgamma(a[i]))

    return ll


def gcp_ln_Phi(double lp, double s, double n, double v, double amin = 1.0):
    return _gcp_ln_Phi(lp, log(s), n, v, amin)


@cython.boundscheck(False)
def gamma_conjugate_prior_bulk_log_p(const double[:] a, const double[:] b,
                                     double lp, double s, double n, double v,
                                     double amin = 1.0):
    """
    Log-probability of the conjugate prior, evaluated in bulk for
    each point `a` and `b` individually. Yields the same results as
    calling `gamma_conjugate_prior_logL` individually for each `a`
    and `b`.

    Returns:
       log_p : Array of logarithms of probabilities for each pair (a[i],b[i]).
    """
    # Sanity:
    cdef size_t N = a.shape[0]
    if N != b.shape[0]:
        raise RuntimeError("Shapes of `a` and `b` do not match.")

    cdef size_t i
    cdef double norm
    cdef double[::1] res = np.empty(N)
    with nogil:
        # Compute the normalization:
        norm = _gcp_ln_Phi(lp, log(s), n, v, amin)

        # Compute the log-likelihood:
        for i in range(N):
            res[i] = ((v*a[i] - 1.0) * log(b[i]) + (a[i] - 1.0) * lp - s*b[i]
                      - n * lgamma(a[i]) - norm)

    return res.base


@cython.boundscheck(False)
def gamma_conjugate_prior_predictive(double[::1] q, double lp, double s,
                                     double n, double v, double amin,
                                     bool inplace=False):
    """
    Posterior predictive of the gamma conjugate prior.
    """
    cdef size_t N = q.shape[0]
    cdef double lnPhi
    cdef double[::1] out
    if inplace:
        out = q
    else:
        out = np.empty(N)

    cdef size_t i
    with nogil:
        GammaConjugatePriorBase\
            .posterior_predictive_pdf(N, &q[0], &out[0], lp, s, n, v, amin, 0.0,
                                      1e-10)

    return out.base


@cython.boundscheck(False)
def gamma_conjugate_prior_predictive_batch(const double[::1] q,
        const double[::1] lp, const double[::1] s, const double[::1] n,
        const double[::1] v, double amin, double epsabs = 0.0,
        double epsrel = 1e-10, out = None):
    # Sanity:
    cdef size_t N = q.shape[0]
    cdef size_t M = lp.shape[0]
    if s.shape[0] != M:
        raise RuntimeError("`lp` and `s` need to be of same shape.")
    if n.shape[0] != M:
        raise RuntimeError("`lp` and `n` need to be of same shape.")
    if v.shape[0] != M:
        raise RuntimeError("`lp` and `v` need to be of same shape.")

    cdef double[:,::1] out_buffer
    if out is None:
        out_buffer = np.empty((M,N))
    else:
        out_buffer = out

    cdef double* out_ptr = &out_buffer[0,0]

    with nogil:
        GammaConjugatePriorBase\
            .posterior_predictive_pdf_batch(N, &q[0], out_ptr, M, &lp[0], &s[0],
                                            &n[0], &v[0], amin, epsabs, epsrel)

    return out.base


@cython.boundscheck(False)
def gamma_conjugate_prior_predictive_cdf(double[::1] q, double lp, double s,
                                         double n, double v, double amin,
                                         double epsabs = 0.0,
                                         double epsrel = 1e-10,
                                         bool inplace=False):
    """
    Posterior predictive of the gamma conjugate prior.
    """
    cdef size_t N = q.shape[0]
    cdef double[::1] out
    if inplace:
        out = q
    else:
        out = np.empty(N)

    with nogil:
        GammaConjugatePriorBase\
            .posterior_predictive_cdf(N, &q[0], &out[0], lp, s, n, v, amin,
                                      epsabs, epsrel)

    return out.base


@cython.boundscheck(False)
def gamma_conjugate_prior_predictive_cdf_batch(const double[::1] q,
        const double[::1] lp, const double[::1] s, const double[::1] n,
        const double[::1] v, double amin, double epsabs = 0.0,
        double epsrel = 1e-10, out = None):
    """
    Posterior predictive of the gamma conjugate prior.
    """
    # Sanity:
    cdef size_t N = q.shape[0]
    cdef size_t M = lp.shape[0]
    if s.shape[0] != M:
        raise RuntimeError("`lp` and `s` need to be of same shape.")
    if n.shape[0] != M:
        raise RuntimeError("`lp` and `n` need to be of same shape.")
    if v.shape[0] != M:
        raise RuntimeError("`lp` and `v` need to be of same shape.")

    cdef double[:,::1] out_buffer
    if out is None:
        out_buffer = np.empty((M,N))
    else:
        out_buffer = out

    cdef double* out_ptr = &out_buffer[0,0]

    with nogil:
        GammaConjugatePriorBase\
            .posterior_predictive_cdf_batch(N, &q[0], out_ptr, M, &lp[0], &s[0],
                                            &n[0], &v[0], amin, epsabs, epsrel)

    return out.base


def gamma_conjugate_prior_kullback_leibler(double lp, double s, double n,
                                           double v, double lp_ref,
                                           double s_ref, double n_ref,
                                           double v_ref, double amin = 1.0,
                                           double epsabs = 0.0,
                                           double epsrel = 1e-10):
    """
    Compute the Kullback-Leibler divergence between a reference
    gamma conjugate prior and another one.
    """
    return GammaConjugatePriorBase.kullback_leibler(lp, s, n, v, lp_ref, s_ref,
                                   n_ref, v_ref, amin, epsabs, epsrel)


@cython.boundscheck(False)
def gamma_mle(const double[:] x, double amin = 1.0):
    """
    Subroutine to compute the Gamma distribution MLE for a set
    of unweighted data.
    """
    cdef size_t i
    cdef size_t N = x.shape[0]

    cdef double xm = 0.0
    cdef double lxm = 0.0
    cdef gamma_mle_t mle
    with nogil:
        # Compute the data statistics:
        for i in range(N):
            xm += x[i]
            lxm += log(x[i])
        xm /= N
        lxm /= N

        mle = compute_gamma_mle(xm, lxm, amin)

    return mle.a, mle.b


def _log_gamma(double x):
    """
    Test exposition of the gamma function in use.
    """
    return _log_gamma_function(x)