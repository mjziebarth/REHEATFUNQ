# The heat flow anomaly corrected gamma conjugate posterior
# of Ziebarth & von Specht (2023) [1].
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ,
#               2022-2023 Malte J. Ziebarth
#
# [1] Ziebarth, M. J. and von Specht, S.: REHEATFUNQ 1.4.0: A model for regional
#     aggregate heat flow distributions and anomaly quantification,
#     EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-222, 2023.
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
from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from cython.operator cimport dereference as deref

from numpy._typing import NDArray

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

cdef extern from "anomaly/variableprecisionposterior.hpp" \
         namespace "reheatfunq::anomaly" nogil:

    # Actually in anomaly/posterior.hpp:
    cdef enum pdf_algorithm_t:
        EXPLICIT = 0
        BARYCENTRIC_LAGRANGE = 1
        ADAPTIVE_SIMPSON = 2

    cdef struct qc_t "reheatfunq::anomaly::posterior::qc_t":
        qc_t(double q, double c)

    cdef struct weighted_sample_t:
        vector[qc_t] sample
        weighted_sample_t(double w)

    cdef cppclass VariablePrecisionPosterior:
        VariablePrecisionPosterior(
                  const vector[weighted_sample_t] weighted_samples,
                  double p, double s, double n, double v, double amin,
                  double dest_tol, precision_t precision,
                  pdf_algorithm_t pa
        ) except+

        double get_Qmax() const

        void pdf_inplace(vector[double]& PH, bool parallel) except+
        void cdf_inplace(vector[double]& PH, bool parallel) except+
        void tail_inplace(vector[double]& PH, bool parallel) except+
        void tail_quantiles_inplace(vector[double]& quantiles,
                                    bool parallel) except+

        bool validate(const vector[vector[qc_t]]& qc_set, double p0, double s0,
                      double n0, double v0, double dest_tol) except+

        void get_locals(size_t l, double& lp, double& ls, double& n, double& v,
	                    double& amin, double& Qmax, vector[double]& ki,
	                    double& h0, double& h1, double& h2, double& h3,
	                    double& w, double& lh0, double& l1p_w, double& log_scale,
                        double& ymax, double& norm) except+

        void get_C(double a, size_t l, double& C0, double& C1, double& C2,
                   double& C3) except+


#
#
#
cdef extern from * namespace "reheatfunq::dirtyhacks" nogil:
    """
    namespace reheatfunq {
    namespace dirtyhacks {
    typedef reheatfunq::anomaly::weighted_sample_t weighted_sample_t;
    typedef reheatfunq::anomaly::posterior::qc_t qc_t;

    void emplace_back(std::vector<weighted_sample_t>& vec, double w)
    {
        vec.emplace_back(w);
    }

    void emplace_back(weighted_sample_t& ws, double q, double c)
    {
        ws.sample.emplace_back(q,c);
    }

    void emplace_back(std::vector<qc_t>& v, double q, double c)
    {
        v.emplace_back(q,c);
    }

    }
    }
    """
    cdef void emplace_back(vector[weighted_sample_t]&, double w)
    cdef void emplace_back(weighted_sample_t& ws, double q, double c)
    cdef void emplace_back(vector[qc_t]& v, double q, double c)


@cython.boundscheck(False)
cdef class CppAnomalyPosterior:
    cdef shared_ptr[VariablePrecisionPosterior] post

    def __init__(self, list qij, list cij, const double[::1] wi, double p,
                 double s, double n, double v, double amin, double rtol,
                 bool validate = False, str pdf_algorithm="barycentric_lagrange",
                 size_t bli_max_splits = 100, uint8_t bli_max_refinements=7,
                 str precision = "long double"):

        pdf_algorithm = pdf_algorithm.lower()
        if pdf_algorithm not in ("explicit", "barycentric_lagrange",
                                 "adaptive_simpson"):
            raise ValueError("`pdf_algorithm` must be one of 'explicit', "
                             "'barycentric_lagrange', or 'adaptive_simpson'.")
        cdef pdf_algorithm_t pa
        if pdf_algorithm == "explicit":
            pa = EXPLICIT
        elif pdf_algorithm == "barycentric_lagrange":
            pa = BARYCENTRIC_LAGRANGE
        elif pdf_algorithm == "adaptive_simpson":
            pa = ADAPTIVE_SIMPSON
        else:
            raise ValueError("'pdf_algorithm' must be one of 'explicit', 'barycentric_lagrange', "
                             "'adaptive_simpson'.")

        cdef size_t M = len(qij)
        if len(cij) != M:
            raise RuntimeError("Length of `cij` and `qij` do not match.")
        if wi.shape[0] != M:
            raise RuntimeError("Length of `wi` and `qij` do not match.")
        cdef vector[weighted_sample_t] weighted_samples
        cdef precision_t prec

        cdef const double[::1] qj
        cdef const double[::1] cj
        cdef size_t i, j, N
        for i in range(M):
            cj = cij[i]
            qj = qij[i]
            N = cj.shape[0]
            if qj.shape[0] != N:
                raise RuntimeError("Length of `cij` and `qij` do not match in "
                                   "element " + str(int(i)) + ".")

            # New weighted sample:
            emplace_back(weighted_samples, wi[i])
            with nogil:
                for j in range(N):
                    emplace_back(weighted_samples.back(), qj[j], cj[j])


        # Parse the numerical precision parameter:
        if prec == "double":
            prec = WP_LONG_DOUBLE
        elif prec == "long double":
            prec = WP_DOUBLE
        elif prec == "float128":
            if not HAS_FLOAT128:
                raise RuntimeError("To use float128 backend, REHEATFUNQ needs to be "
                                   "compiled with the anomaly_posterior_float128 option.")
            prec = WP_FLOAT_128
        elif prec == "dec50":
            if not HAS_DEC50:
                raise RuntimeError("To use float128 backend, REHEATFUNQ needs to be "
                                   "compiled with the anomaly_posterior_float128 option.")
            prec = WP_BOOST_DEC_50
        elif prec == "dec100":
            if not HAS_DEC100:
                raise RuntimeError("To use float128 backend, REHEATFUNQ needs to be "
                                   "compiled with the anomaly_posterior_float128 option.")
            prec = WP_BOOST_DEC_100
        else:
            raise RuntimeError("Unknown precision '" + prec + "'.")


        # Compute the posterior:
        self.post = make_shared[VariablePrecisionPosterior](
                        weighted_samples, p, s, n, v, amin, rtol, prec, pa,
                        bli_max_splits, bli_max_refinements
        )

        # Validate:
        cdef vector[vector[qc_t]] setofsets
        if validate:
            weighted_samples.clear()
            setofsets.resize(M)
            for i in range(M):
                cj = cij[i]
                qj = qij[i]
                N = cj.shape[0]
                with nogil:
                    for j in range(N):
                        emplace_back(setofsets[i], qj[j], cj[j])

            deref(self.post).validate(setofsets, p, s, n, v, rtol)

    #
    # Properties:
    #
    def Qmax(self) -> float:
        if not self.post:
            raise RuntimeError("Not properly initialized.")
        return deref(self.post).get_Qmax()


    def pdf(self, const double[:] PH, bool parallel=True):
        """

        """
        if not self.post:
            raise RuntimeError("Not properly initialized.")
        cdef size_t N = PH.shape[0]
        cdef vector[double] work
        cdef size_t i
        cdef double[::1] res
        work.resize(N)
        with nogil:
            for i in range(N):
                work[i] = PH[i]

            deref(self.post).pdf_inplace(work, parallel)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base


    def cdf(self, const double[:] PH, bool parallel=True):
        """

        """
        if not self.post:
            raise RuntimeError("Not properly initialized.")
        cdef size_t N = PH.shape[0]
        cdef vector[double] work
        cdef size_t i
        cdef double[::1] res
        work.resize(N)
        with nogil:
            for i in range(N):
                work[i] = PH[i]

            deref(self.post).cdf_inplace(work, parallel)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base


    def tail(self, const double[:] PH, bool parallel=True):
        """

        """
        if not self.post:
            raise RuntimeError("Not properly initialized.")
        cdef size_t N = PH.shape[0]
        cdef vector[double] work
        cdef size_t i
        cdef double[::1] res
        work.resize(N)
        with nogil:
            for i in range(N):
                work[i] = PH[i]

            deref(self.post).tail_inplace(work, parallel)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base


    def tail_quantiles(self, const double[:] t, bool parallel=True):
        """

        """
        if not self.post:
            raise RuntimeError("Not properly initialized.")
        cdef size_t N = t.shape[0]
        cdef vector[double] work
        cdef size_t i
        cdef double[::1] res
        work.resize(N)
        with nogil:
            for i in range(N):
                work[i] = t[i]

            deref(self.post).tail_quantiles_inplace(work, parallel)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base

    def _get_locals(self, size_t l):
        """

        """
        if not self.post:
            raise RuntimeError("Not properly initialized.")

        cdef double lp, ls, n, v, amin, Qmax, h0, h1, h2, h3, w, lh0, l1p_w, \
                    log_scale, ymax, norm
        # This line prevents an unecessary Cython compiler warning:
        lp = ls = n = v = amin = Qmax = h0 = h1 = h2 = h3 = w = lh0 = l1p_w \
           = log_scale = ymax = norm = 0.0
        cdef vector[double] ki
        with nogil:
            deref(self.post).get_locals(l, lp, ls, n, v, amin, Qmax, ki, h0,
                                        h1, h2, h3, w, lh0, l1p_w, log_scale,
                                        ymax, norm)

        # Transfer to numpy array:
        cdef double[::1] ki_np = np.zeros(ki.size())
        cdef size_t i
        with nogil:
            for i in range(ki.size()):
                ki_np[i] = ki[i]

        # Transfer to data class:
        from dataclasses import dataclass
        @dataclass
        class Locals:
            lp: float
            ls: float
            n: float
            v: float
            amin: float
            Qmax: float
            ki: NDArray[np.float64]
            h0: float
            h1: float
            h2: float
            h3: float
            w: float
            lh0: float
            l1p_w: float
            log_scale: float
            ymax: float
            norm: float


        return Locals(lp, ls, n, v, amin, Qmax, ki_np, h0, h1, h2, h3, w, lh0, l1p_w,
                      log_scale, ymax, norm)




    def _get_C(self, double a, size_t l):
        """

        """
        if not self.post:
            raise RuntimeError("Not properly initialized.")
        cdef double[:] C = np.empty(4)
        with nogil:
            deref(self.post).get_C(a, l, C[0], C[1], C[2], C[3]);

        return C.base