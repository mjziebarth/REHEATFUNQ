/*
 * Conjugate prior of the gamma distribution due to Miller (1980).
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2021 Deutsches GeoForschungsZentrum GFZ,
 *               2022-2023 Malte J. Ziebarth
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
 *
 * Miller (1980):
 */

#include <cstddef>
#include <memory>
#include <funccache.hpp>
#include <string>

#ifndef PDTOOLBOX_GAMMA_CONJUGATE_PRIOR_HPP
#define PDTOOLBOX_GAMMA_CONJUGATE_PRIOR_HPP

namespace pdtoolbox {

enum condition_policy_t {
	CONDITION_WARN=0, CONDITION_ERROR=1, CONDITION_INF=2
};

class GammaConjugatePriorBase
{
public:
	GammaConjugatePriorBase(double lp, double s, double n, double v,
	                        double amin = 1.0, double epsabs=0,
	                        double epsrel=1e-10);

	/*
	 * Named parameter access:
	 */
	double lp() const;
	double p() const;
	double s() const;
	double n() const;
	double v() const;

	/*
	 * Static methods:
	 */
	static double ln_Phi(double lp, double ls, double n, double v,
	                     double amin, double epsabs, double epsrel=1e-10);

	static double
	kullback_leibler(double lp, double s, double n, double v,
	                 double lp_ref, double s_ref, double n_ref,
	                 double v_ref, double amin, double epsabs,
	                 double epsrel=1e-10,
	                 condition_policy_t on_condition_large=CONDITION_WARN);

	static double
	kullback_leibler_batch_max(const double* params, size_t N, double lp_ref,
	                           double s_ref, double n_ref, double v_ref,
	                           double amin, double epsabs, double epsrel=1e-10,
	                           const long double* ln_Phis = nullptr);

	/*
	 * Helper routines to setup for cached evaluation of the minimum surprise
	 * estimate cost function for SHGO.
	 */
	static std::shared_ptr<SortedCache<4,double>>
	generate_mse_cost_function_cache(const double* params, size_t N,
	                                 double amin, double epsabs,
	                                 double epsrel=1e-10);

	static std::shared_ptr<SortedCache<4,double>>
	restore_mse_cost_function_cache(const double* cache_dump, size_t M,
	                                const double* params, size_t N,
	                                double amin, double epsabs,
	                                double epsrel=1e-10);

	static void
	posterior_predictive_pdf(const size_t Nq, const double* q,
	                         double* out,
	                         double lp, double s, double n, double v,
	                         double amin, double epsabs, double epsrel,
	                         bool parallel=true);

	static void
	posterior_predictive_pdf_batch(const size_t Nq, const double* q,
	                               double* out, const size_t Mparam,
	                               const double* lp, const double* s,
	                               const double* n, const double* v,
	                               double amin, double epsabs, double epsrel,
	                               bool parallel=true);

	static int
	posterior_predictive_pdf_common_norm(const size_t Nq, const double* q,
	                               double* out, const size_t Mparam,
	                               const double* lp, const double* s,
	                               const double* n, const double* v,
	                               double amin, double epsabs, double epsrel,
	                               std::string& err_msg,
	                               bool parallel=true);

	static void
	posterior_predictive_cdf(const size_t Nq, const double* q,
	                         double* out,
	                         double lp, double s, double n, double v,
	                         double amin, double epsabs, double epsrel);

	static void
	posterior_predictive_cdf_batch(const size_t Nq, const double* q,
	                               double* out, const size_t Mparam,
	                               const double* lp, const double* s,
	                               const double* n, const double* v,
	                               double amin, double epsabs, double epsrel);

	static int
	posterior_predictive_cdf_common_norm(
	        const size_t Nq, const double* q, double* out,const size_t Mparam,
	        const double* lp, const double* s, const double* n, const double* v,
	        double amin, double epsabs, double epsrel,
	        std::string& err_msg, bool parallel=true);


protected:
	const double amin;
	const double epsrel, epsabs;
	double lp_, p_, ls, s_, v_, n_;
	double lPhi;


private:
	static void
	posterior_predictive_pdf(const size_t Nq, const double* q,
	                         double* out,
	                         double lp, double s, double n, double v,
	                         double amin, double epsabs, double epsrel,
	                         bool parallel, double ln_Phi, double ls);

	struct lnPhi_max_t {
		double lnPhi_max;
		double tot_norm;
	};

	static lnPhi_max_t compute_lnPhi_max(
	    const size_t Mparam, const double* lp, const double* s,
	    const double* n, const double* v, double amin,
	    double epsabs, double epsrel, bool& error_flag,
	    std::string& err_msg, bool parallel
	);


};


}

#endif