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
from scipy.optimize import shgo
from libc.math cimport log, sqrt, fabs, exp, pow, ceil, log1p, isnan, lgamma
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr, shared_ptr, make_shared
from cython.operator cimport dereference as deref
from cython.parallel cimport prange

cdef extern from "funccache.hpp" namespace "pdtoolbox" nogil:
    """
    namespace pdtoolbox {

    typedef SortedCache<4,double> GCPCache;

    }
    """
    cdef cppclass array4d "std::array<double,4>":
        array4d()
        double operator[](size_t i) const
        double& operator[](size_t i)
        double& operator[](int i)

    cdef cppclass GCPCache:
        double operator()(const array4d&) const
        size_t hits() const
        size_t misses() const
        double hit_rate() const
        vector[pair[const array4d, double]] dump() const
        bool operator==(const GCPCache& other) const
        bool operator!=(const GCPCache& other) const
        size_t size() const

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
    cdef enum condition_policy_t:
        CONDITION_WARN=0,
        CONDITION_ERROR=1,
        CONDITION_INF=2

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
                                double epsrel,
                                int on_condition_large) except+

        @staticmethod
        double kullback_leibler_batch_max(const double* params, size_t N,
                                          double lp_ref, double s_ref,
                                          double n_ref, double v_ref,
                                          double amin, double epsabs,
                                          double epsrel) except+

        @staticmethod
        shared_ptr[GCPCache] \
        generate_mse_cost_function_cache(const double* params, size_t N,
                                         double amin, double epsabs,
                                         double epsrel)

        @staticmethod
        shared_ptr[GCPCache] \
        restore_mse_cost_function_cache(const double* cache_dump, size_t M,
                                        const double* params, size_t N,
                                        double amin, double epsabs,
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
        int posterior_predictive_pdf_common_norm(size_t Nq, const double* q,
                                double* out, size_t Mparam, const double* lp,
                                const double* s, const double* n,
                                const double* v, double amin, double epsabs,
                                double epsrel, string& err_msg)

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

        @staticmethod
        int posterior_predictive_cdf_common_norm(
            const size_t Nq, const double* q, double* out,const size_t Mparam,
            const double* lp, const double* s, const double* n, const double* v,
            double amin, double epsabs, double epsrel,
            string& err_msg
        )

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
def gamma_conjugate_prior_predictive_common_norm(const double[::1] q,
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

    cdef double[::1] out_buffer
    if out is None:
        out_buffer = np.empty(N)
    else:
        out_buffer = out

    cdef double* out_ptr = &out_buffer[0]
    cdef string err_msg
    cdef int code

    with nogil:
        code = GammaConjugatePriorBase\
            .posterior_predictive_pdf_common_norm(
                N, &q[0], out_ptr, M, &lp[0], &s[0],
                &n[0], &v[0], amin, epsabs, epsrel, err_msg)

    if code != 0:
        raise RuntimeError(str(err_msg))

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


@cython.boundscheck(False)
def gamma_conjugate_prior_predictive_cdf_common_norm(const double[::1] q,
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

    cdef double[::1] out_buffer
    if out is None:
        out_buffer = np.empty(N)
    else:
        out_buffer = out

    cdef double* out_ptr = &out_buffer[0]

    cdef string err_msg
    cdef int code

    with nogil:
        code = GammaConjugatePriorBase\
            .posterior_predictive_cdf_common_norm(
                N, &q[0], out_ptr, M, &lp[0], &s[0],
                &n[0], &v[0], amin, epsabs, epsrel, err_msg)

    if code != 0:
        raise RuntimeError(str(err_msg))

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
                                   n_ref, v_ref, amin, epsabs, epsrel,
                                   CONDITION_ERROR)


@cython.boundscheck(False)
def gamma_conjugate_prior_kullback_leibler_batch_max(
        const double[:,::1] params, double lp_ref, double s_ref, double n_ref,
        double v_ref, double amin = 1.0, double epsabs = 0.0,
        double epsrel = 1e-10
    ) -> float:
    """
    Compute the maximum Kullback-Leibler divergence between a reference
    gamma conjugate prior and multiple others.
    """
    cdef size_t N = params.shape[0]
    if N == 0:
        raise RuntimeError("Need at least one parameter tuple in `params`.")
    if params.shape[1] != 4:
        raise RuntimeError("Shape of `params` needs to be (N,4).")
    return GammaConjugatePriorBase.kullback_leibler_batch_max(&params[0,0], N,
                           lp_ref, s_ref, n_ref, v_ref, amin, epsabs, epsrel)


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


cdef class GCPMSECache:
    """
    Cache for the GammaConjugatePrior.minimum_surprise_estimate method.
    """
    cdef shared_ptr[GCPCache] _cache
    cdef shared_ptr[vector[double]] _updated_params
    cdef double _amin

    def __init__(self, const double[:,::1] gcp_updated_params, double amin):
        state = (None, gcp_updated_params, amin)
        self.__setstate__(state)

    def __call__(self, const double lp, const double s, const double n,
                 const double v):
        if not self._cache:
            raise RuntimeError("Cache not initialized.")
        cdef array4d xcpp
        xcpp[0] = lp
        xcpp[1] = s
        xcpp[2] = n
        xcpp[3] = v

        cdef double KL = np.inf
        with nogil:
            KL = deref(self._cache)(xcpp)

        return KL

    @cython.boundscheck(False)
    def __setstate__(self, tuple state):
        cdef const double[:,::1] gcp_updated_params = state[1]
        cdef double amin = state[2]
        cdef double[::1] cache_npy

        # Sanity:
        if gcp_updated_params.shape[1] != 4:
            raise RuntimeError("Shape of `gcp_updated_params` needs to be "
                               "(n,4).")

        # Copy the data to a C++ vector that is owned by this object:
        cdef size_t n = gcp_updated_params.shape[0]
        cdef size_t N = 4 * n
        self._updated_params = make_shared[vector[double]](N)
        cdef size_t i,j,k
        with nogil:
            for i in range(n):
                for j in range(4):
                    deref(self._updated_params)[4*i+j] \
                       = gcp_updated_params[i,j]

        # Parameters:
        self._amin = amin

        # Initialize the cache:
        if state[0] is None:
            # New cache:
            self._cache = \
               GammaConjugatePriorBase.generate_mse_cost_function_cache(
                          &deref(self._updated_params)[0], n, amin,
                          0.0, 1e-10
               )
        else:
            cache_npy = state[0]
            if (cache_npy.shape[0] % 5) != 0:
                raise RuntimeError("Shape of cache dump needs to conform to "
                                   "(n,5) reshape.")
            self._cache = \
               GammaConjugatePriorBase.restore_mse_cost_function_cache(
                          &cache_npy[0], cache_npy.shape[0],
                          &deref(self._updated_params)[0], n, amin,
                          0.0, 1e-10
               )

    @staticmethod
    cdef GCPMSECache empty():
        return GCPMSECache.__new__(GCPMSECache)

    def __eq__(self, other) -> bool:
        if not isinstance(other, GCPMSECache):
            return False
        cdef GCPMSECache c2 = other
        cdef size_t i,j,n0,n1
        cdef double x1, x2
        if self._amin != c2._amin:
            return False

        # Ensure that the updated parameters are equal:
        if self._updated_params and c2._updated_params:
            n1 = deref(self._updated_params).size()
            n2 = deref(c2._updated_params).size()
            if n1 != n2:
                return False
            with nogil:
                for i in range(n1):
                    x1 = deref(self._updated_params)[i]
                    x2 = deref(c2._updated_params)[i]
                    if x1 != x2:
                        with gil:
                            return False
        elif self._updated_params or c2._updated_params:
            return False

        # Ensure that the caches are equal:
        if self._cache and c2._cache:
            if deref(self._cache) != deref(c2._cache):
                return False
        elif self._cache or c2._cache:
            return False

        return True

    def hits(self) -> int:
        """
        Cache hits.
        """
        if not self._cache:
            return 0
        return deref(self._cache).hits()

    def misses(self) -> int:
        """
        Cache misses.
        """
        if not self._cache:
            return 0
        return deref(self._cache).misses()

    def hit_rate(self) -> float:
        """
        Cache hit rate.
        """
        if not self._cache:
            return 0
        return deref(self._cache).hit_rate()

    def size(self) -> int:
        if not self._cache:
            return 0
        return deref(self._cache).size()

    @cython.boundscheck(False)
    def __reduce__(self) -> tuple:
        """
        Pickling support.
        """
        # Get the cache state:
        cdef vector[pair[const array4d, double]] cache_dump
        cdef double[::1] cache_npy_view
        cdef double[:,::1] up_view
        cdef size_t i, j, n_up
        if self._cache and self._updated_params:
            cache_dump = deref(self._cache).dump()
            cache_npy = np.empty(5 * cache_dump.size())
            n_up = deref(self._updated_params).size() // 4
            gcp_updated_params = np.empty((n_up,4))
            cache_npy_view = cache_npy
            up_view = gcp_updated_params
            with nogil:
                for i in range(cache_dump.size()):
                    cache_npy_view[5*i]   = cache_dump[i].first[0]
                    cache_npy_view[5*i+1] = cache_dump[i].first[1]
                    cache_npy_view[5*i+2] = cache_dump[i].first[2]
                    cache_npy_view[5*i+3] = cache_dump[i].first[3]
                    cache_npy_view[5*i+4] = cache_dump[i].second
                for i in range(n_up):
                    for j in range(4):
                        up_view[i,j] = deref(self._updated_params)[4*i+j]
        else:
            cache_npy = None

        state = (cache_npy, gcp_updated_params, self._amin)
        return (_gen_empty_GCPMSECache_for_pickling, (), state, None, None)



def _gen_empty_GCPMSECache_for_pickling() -> GCPMSECache:
    return GCPMSECache.empty()

def gamma_conjugate_prior_minimum_surprise_backend(
        const double[:,::1] gcp_updated_params, double amin, tuple bounds,
        dict kwargs, bool verbose, GCPMSECache cache = GCPMSECache.empty()
    ):
    """
    Backend of the `minimum_surprise_estimate` function of the
    `GammaConjugatePrior` class.
    """
    if gcp_updated_params.shape[1] != 4:
        raise RuntimeError("`gcp_updated_params` has to be of shape (n,4).")
    if gcp_updated_params.shape[0] == 0:
        raise RuntimeError("`gcp_updated_params` is empty.")

    cdef double lpmin = bounds[0][0]
    cdef double lpmax = bounds[0][1]
    cdef double smin = bounds[1][0]
    cdef double smax = bounds[1][1]
    cdef double lnvspmin = bounds[2][0]
    cdef double lnvspmax = bounds[2][1]
    cdef double vmin = bounds[3][0]
    cdef double vmax = bounds[3][1]

    # See whether we have a cache or not:
    cdef shared_ptr[GCPCache] cache_cpp
    if not cache._cache:
        cache_cpp = GammaConjugatePriorBase.generate_mse_cost_function_cache(
                        &gcp_updated_params[0,0], gcp_updated_params.shape[0],
                        amin, 0.0, 1e-10)
    else:
        cache_cpp = cache._cache

    @cython.boundscheck(False)
    def cost(const double[::1] x) -> double:
        # Check bounds.
        # We need to do this because the local minimization step performed
        # by SHGO might be out of bounds (2023-08-04). In that case,
        # return infinity.
        if x[0] < lpmin or x[0] > lpmax or x[1] < smin or x[1] > smax \
                 or x[3] < vmin or x[3] > vmax or x[2] < lnvspmin \
                 or x[2] > lnvspmax:
            return np.inf

        # Retrieve parameters:
        cdef array4d xcpp
        xcpp[0] = x[0] # lp
        xcpp[1] = x[1] # s
        xcpp[3] = x[3] # v
        xcpp[2] = x[3] * (1.0 + exp(x[2])) # n

        cdef double KL = np.inf
        with nogil:
            KL = deref(cache_cpp)(xcpp)

        return KL

    # Perform the optimization:
    res = shgo(cost, bounds=bounds, **kwargs)

    return res