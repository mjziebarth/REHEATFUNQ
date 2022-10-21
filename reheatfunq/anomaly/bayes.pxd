# distutils: language = c++
#
# The heat flow anomaly corrected gamma conjugate posterior
# of Ziebarth et al. (2022) [1].
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ
#
# [1] TODO
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

cdef extern from * nogil:
    """
    void marginal_posterior_c(const double* x, double* res, const size_t Nx,
                              double p, double s, double n, double nu,
                              const double* qi, const double* ci,
                              const size_t N, const size_t workspace_size,
                              double dest_tol);

    enum posterior_t {
        DENSITY, CUMULATIVE, TAIL, UNNORMED_LOG
    };

    void posterior(const double* x, double* res, size_t Nx, const double* qi,
                   const double* ci, size_t N, double p, double s, double n,
                   double nu, double dest_tol, posterior_t type);

    void posterior_pdf(const double* x, double* res, size_t Nx,
                       const double* qi, const double* ci, size_t N, double p,
                       double s, double n, double nu,double dest_tol);

    void posterior_cdf(const double* x, double* res, size_t Nx,
                       const double* qi, const double* ci, size_t N, double p,
                       double s, double n, double nu,double dest_tol);

    void posterior_tail(const double* x, double* res, size_t Nx,
                        const double* qi, const double* ci, size_t N, double p,
                        double s, double n, double nu, double dest_tol);

    void posterior_log_unnormed(const double* x, double* res, size_t Nx,
                                const double* qi, const double* ci, size_t N,
                                double p, double s, double n, double nu,
                                double dest_tol);


    void posterior_silent(const double* x, double* res, size_t Nx,
                  const double* qi, const double* ci, size_t N, double p,
                  double s, double n, double nu, double dest_tol,
                  posterior_t type);

    void log_posterior_debug(const double* x, const double* a, double* res,
                             size_t Nax, const double* qi, const double* ci,
                             const size_t N, const double p, const double s,
                             const double n, const double nu, double dest_tol
                             );

    int tail_quantiles_intcode(const double* quantiles, double* res,
                           const size_t Nquant, const double* qi,
                           const double* ci, const size_t N, const double p,
                           const double s, const double n, const double nu,
                           const double dest_tol, short print_to_cerr);
    """
    cdef void marginal_posterior_c(const double* x, double* res,
                                   const size_t Nx, double p, double s,
                                   double n, double nu, const double* qi,
                                   const double* ci, const size_t N,
                                   double dest_tol) except+

    ctypedef enum posterior_t: DENSITY, CUMULATIVE, TAIL, UNNORMED_LOG

    cdef void posterior_pdf(const double* x, double* res, size_t Nx,
                            const double* qi, const double* ci, size_t N,
                            double p, double s, double n, double nu,
                            double dest_tol) except+

    cdef void posterior_cdf(const double* x, double* res, size_t Nx,
                            const double* qi, const double* ci, size_t N,
                            double p, double s, double n, double nu,
                            double dest_tol) except+

    cdef void posterior_tail(const double* x, double* res, size_t Nx,
                             const double* qi, const double* ci, size_t N,
                             double p, double s, double n, double nu,
                             double dest_tol) except+

    cdef void posterior_log_unnormed(const double* x, double* res, size_t Nx,
                                     const double* qi, const double* ci,
                                     size_t N, double p, double s, double n,
                                     double nu, double dest_tol
                                     ) except+

    # A version of the above inserting NaNs on fail:
    cdef void posterior_silent(const double* x, double* res, size_t Nx,
                        const double* qi, const double* ci, size_t N,
                        double p, double s, double n, double nu,
                        double dest_tol,
                        posterior_t type)

    cdef void log_posterior_debug(const double* x, const double* a, double* res,
                         size_t Nax, const double* qi, const double* ci,
                         const size_t N, const double p, const double s,
                         const double n, const double nu,
                         double dest_tol)

    cdef int tail_quantiles_intcode(const double* quantiles, double* res,
                        const size_t Nquant, const double* qi, const double* ci,
                        const size_t N, const double p, const double s,
                        const double n, const double nu,
                        const double dest_tol, short print_to_cerr)
