/*
 * Heat flow anomaly analysis posterior numerics.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2021-2022 Deutsches GeoForschungsZentrum GFZ,
 *                    2022 Malte J. Ziebarth
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

enum precision_t {
	WP_DOUBLE = 0,
	WP_LONG_DOUBLE = 1,
	WP_FLOAT_128 = 2,
	WP_BOOST_DEC_50 = 3,
	WP_BOOST_DEC_100 = 4
};

void posterior_pdf(const double* x, long double* res, size_t Nx,
                   const double* qi, const double* ci, size_t N, double p,
                   double s, double n, double nu, double amin, double dest_tol,
                   precision_t working_precision);

void posterior_pdf_batch(const double* x, size_t Nx, long double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double amin, double dest_tol,
                         precision_t working_precision);

void posterior_cdf(const double* x, long double* res, size_t Nx,
                   const double* qi, const double* ci, size_t N, double p,
                   double s, double n, double nu, double amin, double dest_tol,
                   precision_t working_precision);

void posterior_cdf_batch(const double* x, size_t Nx, long double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double amin, double dest_tol,
                         precision_t working_precision);

void posterior_tail(const double* x, long double* res, size_t Nx,
                    const double* qi, const double* ci, size_t N, double p,
                    double s, double n, double nu, double amin,
                    double dest_tol,
                    precision_t working_precision);

void posterior_tail_batch(const double* x, size_t Nx, long double* res,
                          const std::vector<const double*>& qi,
                          const std::vector<const double*>& ci,
                          const std::vector<size_t>& N,
                          double p, double s, double n, double nu,
                          double amin, double dest_tol,
                          precision_t working_precision);

void posterior_log_unnormed(const double* x, long double* res, size_t Nx,
                            const double* qi, const double* ci, size_t N,
                            double p, double s, double n, double nu,
                            double amin, double dest_tol,
                            precision_t working_precision);

void posterior_silent(const double* x, long double* res, size_t Nx,
                      const double* qi, const double* ci, size_t N, double p,
                      double s, double n, double nu, double amin,
                      double dest_tol, posterior_t type,
                      precision_t working_precision);

void tail_quantiles(const double* quantiles, double* res, const size_t Nquant,
                    const double* qi, const double* ci, const size_t N,
                    const double p, const double s, const double n,
                    const double nu, const double amin, const double dest_tol);

void posterior_tail_quantiles_batch(
                    const double* quantiles, double* res, const size_t Nquant,
                    const std::vector<const double*>& qi,
                    const std::vector<const double*>& ci,
                    const std::vector<size_t>& N,
                    double p, double s, double n, double nu,
                    double amin, double dest_tol);

void posterior_tail_quantiles_batch_barycentric_lagrange(
                    const double* quantiles, double* res, const size_t Nquant,
                    const std::vector<const double*>& qi,
                    const std::vector<const double*>& ci,
                    const std::vector<size_t>& N,
                    double p, double s, double n, double nu,
                    double amin, double dest_tol, precision_t precision,
                    size_t n_chebyshev);

int tail_quantiles_intcode(const double* quantiles, double* res,
                           const size_t Nquant, const double* qi,
                           const double* ci, const size_t N, const double p,
                           const double s, const double n, const double nu,
                           const double amin, const double dest_tol,
                           short print);

}

}

#endif