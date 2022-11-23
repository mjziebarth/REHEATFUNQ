/*
 *
 */

#include <cstddef>
#include <vector>

#ifndef PDTOOLBOX_ZIEBARTH2022A_HPP
#define PDTOOLBOX_ZIEBARTH2022A_HPP

namespace pdtoolbox {

namespace heatflow {

enum posterior_t {
	DENSITY = 0,
	CUMULATIVE = 1,
	TAIL = 2,
	UNNORMED_LOG = 4
};

void posterior_pdf(const double* x, double* res, size_t Nx, const double* qi,
                   const double* ci, size_t N, double p, double s, double n,
                   double nu, double dest_tol);

void posterior_pdf_batch(const double* x, size_t Nx, double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double dest_tol);

void posterior_cdf(const double* x, double* res, size_t Nx, const double* qi,
                   const double* ci, size_t N, double p, double s, double n,
                   double nu, double dest_tol);

void posterior_cdf_batch(const double* x, size_t Nx, double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double dest_tol);

void posterior_tail(const double* x, double* res, size_t Nx, const double* qi,
                    const double* ci, size_t N, double p, double s, double n,
                    double nu, double dest_tol);

void posterior_tail_batch(const double* x, size_t Nx, double* res,
                          const std::vector<const double*>& qi,
                          const std::vector<const double*>& ci,
                          const std::vector<size_t>& N,
                          double p, double s, double n, double nu,
                          double dest_tol);

void posterior_log_unnormed(const double* x, double* res, size_t Nx,
                            const double* qi, const double* ci, size_t N,
                            double p, double s, double n, double nu,
                            double dest_tol);

void posterior_silent(const double* x, double* res, size_t Nx, const double* qi,
               const double* ci, size_t N, double p, double s, double n,
               double nu, double dest_tol, posterior_t type);

void tail_quantiles(const double* quantiles, double* res, const size_t Nquant,
                    const double* qi, const double* ci, const size_t N,
                    const double p, const double s, const double n,
                    const double nu, const double dest_tol);

int tail_quantiles_intcode(const double* quantiles, double* res,
                           const size_t Nquant, const double* qi,
                           const double* ci, const size_t N, const double p,
                           const double s, const double n, const double nu,
                           const double dest_tol,
                           short print);

}

}

#endif