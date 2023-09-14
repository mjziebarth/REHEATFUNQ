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
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from cython.operator cimport dereference as deref

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
    cdef struct qc_t "reheatfunq::anomaly::posterior::qc_t":
        qc_t(double q, double c)

    cdef struct weighted_sample_t:
        vector[qc_t] sample
        weighted_sample_t(double w)

    cdef cppclass VariablePrecisionPosterior:
        VariablePrecisionPosterior(
                  const vector[weighted_sample_t] weighted_samples,
                  double p, double s, double n, double v, double amin,
                  double dest_tol
        ) except+

        double get_Qmax() const

        void pdf_inplace(vector[double]& PH) except+
        void cdf_inplace(vector[double]& PH, bool parallel,
                         bool adaptive) except+
        void tail_inplace(vector[double]& PH, bool parallel,
                          bool adaptive) except+
        void tail_quantiles_inplace(vector[double]& quantiles,
                                    size_t n_chebyshev, bool parallel,
                                    bool adaptive) except+

        bool validate(const vector[vector[qc_t]]& qc_set, double p0, double s0,
                      double n0, double v0, double dest_tol) except+


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
                 double s, double n, double v, double amin, double dest_tol,
                 bool validate = False):

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

        #raise RuntimeError("check 2")


        prec = WP_LONG_DOUBLE

        with nogil:
            self.post = make_shared[VariablePrecisionPosterior](
                            weighted_samples, p, s, n, v, amin, dest_tol, prec
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

            deref(self.post).validate(setofsets, p, s, n, v, dest_tol)

    #
    # Properties:
    #
    def Qmax(self) -> float:
        if not self.post:
            raise RuntimeError("Not properly initialized.")
        return deref(self.post).get_Qmax()


    def pdf(self, const double[:] PH):
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

            deref(self.post).pdf_inplace(work)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base


    def cdf(self, const double[:] PH, bool parallel=True, bool adaptive=False):
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

            deref(self.post).cdf_inplace(work, parallel, adaptive)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base


    def tail(self, const double[:] PH, bool parallel=True, bool adaptive=False):
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

            deref(self.post).tail_inplace(work, parallel, adaptive)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base


    def tail_quantiles(self, const double[:] t, size_t n_chebyshev=100,
                       bool parallel=True, bool adaptive=False):
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

            deref(self.post).tail_quantiles_inplace(work, n_chebyshev, parallel,
                                                    adaptive)
        res = np.empty(N)
        with nogil:
            for i in range(N):
                res[i] = work[i]

        return res.base