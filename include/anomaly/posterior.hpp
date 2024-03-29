/*
 * Code for computing the gamma conjugate posterior modified for heat
 * flow anomaly as described by Ziebarth & von Specht [1].
 * This code is an alternative implementation of the code found in
 * `ziebarth2021a.cpp`.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ,
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
 * [1] Ziebarth, M. J. and von Specht, S.: REHEATFUNQ 1.4.0: A model for
 *     regional aggregate heat flow distributions and anomaly quantification,
 *     EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-222, 2023.
 */

#ifndef REHEATFUNQ_ANOMALY_POSTERIOR_HPP
#define REHEATFUNQ_ANOMALY_POSTERIOR_HPP

/*
 * General includes:
 */
#ifndef BOOST_ENABLE_ASSERT_HANDLER
#define BOOST_ENABLE_ASSERT_HANDLER // Make sure the asserts do not abort
#endif
#include <vector>
#include <algorithm>
#include <memory>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/tools/roots.hpp>

#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <type_traits>

/*
 * REHEATFUNQ includes
 */
#include <anomaly/posterior/localsandnorm.hpp>
#include <numerics/kahan.hpp>
#include <numerics/simpson.hpp>
#include <numerics/barylagrint.hpp>

namespace reheatfunq {
namespace anomaly {

namespace rm = reheatfunq::math;
namespace rn = reheatfunq::numerics;
namespace bmq = boost::math::quadrature;
namespace bmt = boost::math::tools;
/*
 * Types:
 */
struct weighted_sample_t {
	std::vector<posterior::qc_t> sample;
	const double w;

	weighted_sample_t(double w) : w(w) {}

};

enum pdf_algorithm_t {
	EXPLICIT = 0,
	BARYCENTRIC_LAGRANGE = 1,
	ADAPTIVE_SIMPSON = 2
};

template<typename real, pdf_algorithm_t pdf_algorithm=BARYCENTRIC_LAGRANGE>
class Posterior {
public:

	Posterior(const std::vector<weighted_sample_t>& weighted_samples,
	          double p, double s, double n, double v, double amin,
	          double rtol, size_t bli_max_splits,
	          uint8_t bli_max_refinements)
	   : locals(init_locals(weighted_samples, p, s, n, v, amin, rtol)),
	     weights(init_weights(weighted_samples, locals)),
	     p(p), s(s), n(n), v(v), Qmax(global_Qmax(locals)),
	     norm(compute_norm(locals, weights)), rtol(rtol),
	     bli_max_splits(bli_max_splits),
	     bli_max_refinements(bli_max_refinements),
	     tmin(rm::log(std::numeric_limits<double>::epsilon() * Qmax)),
	     tmax(rm::log((1.0 - transition) * Qmax))
	{
		if (pdf_algorithm == BARYCENTRIC_LAGRANGE){
			init_log_pdf_bli(locals, weights, Qmax, norm,
			                 bli_max_splits, bli_max_refinements,
			                 rtol);
		} else if (pdf_algorithm == ADAPTIVE_SIMPSON){
			init_simpson_quadrature();
		}
	}

	double get_Qmax() const {
		return static_cast<double>(Qmax);
	}

	/*
	 * The probability density function (PDF)
	 */
	void pdf(std::vector<double>& PH, bool parallel=true) const
	{
		const size_t Nx = PH.size();
		std::vector<double>& result = PH;

		std::optional<std::exception> except;
		#pragma omp parallel for schedule(guided) if(parallel)
		for (size_t i=0; i<Nx; ++i){
			if (except)
				continue;
			/*
			 * Shortcuts for out-of-bounds:
			 */
			if (PH[i] < 0){
				result[i] = 0.0;
				continue;
			} else if (PH[i] >= Qmax){
				result[i] = 0.0;
				continue;
			}
			try {
				result[i] = static_cast<double>(pdf_single(
					rn::PointInInterval<real>(PH[i], PH[i], Qmax-PH[i])
				));
			} catch (const std::exception& e) {
				except = e;
			}
		}

		if (except)
			throw except;
	}

	template<bool use_saq=true>
	void cdf(std::vector<double>& PH, bool parallel=true) const
	{
		const size_t Nx = PH.size();

		if (!saq)
			init_simpson_quadrature();

		/*
			* OMP-compatible exception propagation code:
			*/
		std::exception_ptr error;

		#pragma omp parallel for if(parallel)
		for (size_t i=0; i<Nx; ++i){
			if (error)
				continue;
			std::exception_ptr local_error;
			if (PH[i] <= 0.0)
				PH[i] = 0.0;
			else if (PH[i] >= Qmax)
				PH[i] = 1.0;
			else {
				try {
					/*
						* The actual numerical code:
						*/
					PH[i] = static_cast<double>(saq->integral(
						rn::PointInInterval<real>(
						    PH[i], PH[i], Qmax - PH[i]
						)
					));
				} catch (...) {
					local_error = std::current_exception();
				}
				if (local_error){
					#pragma omp critical
					{
						error = local_error;
					}
				}
			}
		}

		/* Handle potential errors that occurred: */
		if (error){
			std::string msg = "An error occurred while computing the "
			                  "posterior CDF: ";
			try {
				std::rethrow_exception(error);
			} catch (const std::runtime_error& e) {
				msg += e.what();
			}
			throw std::runtime_error(msg);
		}
	}


	template<typename real_in=double, bool use_saq=true>
	void tail(std::vector<real_in>& PH, bool parallel=true) const
	{
		const size_t Nx = PH.size();

		if (!saq)
			init_simpson_quadrature();

		/*
		* OMP-compatible exception propagation code:
		*/
		std::exception_ptr error;

		#pragma omp parallel for if(parallel)
		for (size_t i=0; i<Nx; ++i){
			if (error)
				continue;
			std::exception_ptr local_error;
			try {
				/*
				 * The actual numerical code:
				 */
				PH[i] = static_cast<double>(saq->integral(
				    rn::PointInInterval<real>(
				        PH[i], PH[i], Qmax - PH[i]
				    ),
				    true
				));
			} catch (...) {
				local_error = std::current_exception();
			}
			if (local_error){
				#pragma omp critical
				{
					error = local_error;
				}
			}
		}

		/* Handle potential errors that occurred: */
		if (error){
			std::string msg = "An error occurred while computing the "
								"posterior tail CDF: ";
			try {
				std::rethrow_exception(error);
			} catch (const std::runtime_error& e) {
				msg += e.what();
			}
			throw std::runtime_error(msg);
		}
	}

	template<bool use_saq=true>
	void tail_quantiles(std::vector<double>& quantiles,
	                    bool parallel=true) const
	{
		if (!saq){
			init_simpson_quadrature();
		}

		/*
		 * OMP-compatible exception propagation code:
		 */
		std::exception_ptr error;

		#pragma omp parallel for if(parallel)
		for (size_t i=0; i<quantiles.size(); ++i){
			if (error)
				continue;

			std::exception_ptr local_error;
			if (quantiles[i] == 0.0)
				quantiles[i] = static_cast<double>(Qmax);
			else if (quantiles[i] == 1.0)
				quantiles[i] = 0.0;
			else if (quantiles[i] > 0.0 && quantiles[i] < 1.0){
				/* The typical case. Use TOMS 748 on a quantile
				 * function to find the quantile.
				 */
				const real qi = quantiles[i];
				auto quantile_function =
				[this,qi](real PH_back) -> real
				{
					const real PH = this->Qmax - PH_back;
					return saq->integral(
					    rn::PointInInterval<real>(PH, PH, PH_back),
					    true
					) - qi;
				};
				std::uintmax_t max_iter(100);
				bmt::eps_tolerance<real>
				    eps_tol(std::numeric_limits<real>::digits - 2);
				try {
					std::pair<real,real> bracket
					    = bmt::toms748_solve(quantile_function,
					                    static_cast<real>(0.0), Qmax,
					                    static_cast<real>(-qi),
					                    static_cast<real>(1.0 - qi),
					                    eps_tol, max_iter);
					quantiles[i] = static_cast<double>(
					    Qmax - 0.5*(bracket.first + bracket.second)
					);
				} catch (...) {
					local_error = std::current_exception();
				}
			} else {
				local_error = std::make_exception_ptr(
				    std::runtime_error("Encountered quantile out of bounds "
				                       "[0,1]."));
			}
			if (local_error){
				#pragma omp critical
				{
					error = local_error;
				}
			}
		}

		/* Handle potential errors that occurred: */
		if (error){
			std::string msg = "An error occurred while computing the "
			                  "posterior tail quantiles: ";
			try {
				std::rethrow_exception(error);
			} catch (const std::runtime_error& e) {
				msg += e.what();
			}
			throw std::runtime_error(msg);
		}
	}


	/*
	 * Add more data:
	 */
	Posterior
	add_samples(const std::vector<weighted_sample_t>& weighted_samples)
	const
	{
		std::vector<posterior::LocalsAndNorm<real>> res_locals(locals);
		std::vector<real> res_weights(weights);

		/* Parameters for the new samples */
		std::vector<posterior::LocalsAndNorm<real>>
		   new_locals(init_locals(weighted_samples, p, s, n, v,
		                          static_cast<double>(locals[0].amin),
		                          rtol));
		std::vector<real>
		   new_weights(init_weights(weighted_samples, new_locals));

		/* Join both: */
		res_locals.insert(res_locals.end(), new_locals.cbegin(),
		                  new_locals.cend());
		res_weights.insert(res_weights.end(), new_weights.cbegin(),
		                   new_weights.cend());

		real res_Qmax(global_Qmax(res_locals));
		real res_norm(compute_norm(res_locals, res_weights));

		return Posterior(std::move(res_locals), std::move(res_weights), p, s,
		                 n, v, static_cast<double>(locals[0].amin), rtol,
		                 res_Qmax, res_norm, bli_max_splits,
		                 bli_max_refinements);
	}


	/*
	 * Validation against the previous implementation; not for general use.
	 */
	bool validate(const std::vector<std::vector<posterior::qc_t>>& qc_set,
	              double p0, double s0, double n0, double v0,
	              double rtol) const;

	void get_locals(size_t l, double& lp, double& ls, double& n, double& v,
	                double& amin, double& Qmax, std::vector<double>& ki,
	                double& h0, double& h1, double& h2, double& h3,
	                double& w, double& lh0, double& l1p_w, double& log_scale,
					double& ymax, double& norm) const
	{
		if (l >= locals.size())
			throw std::runtime_error("Index 'l' out of bounds.");
		lp = static_cast<double>(locals[l].lp);
		ls = static_cast<double>(locals[l].ls);
		n = static_cast<double>(locals[l].n);
		v = static_cast<double>(locals[l].v);
		amin = static_cast<double>(locals[l].amin);
		Qmax = static_cast<double>(locals[l].Qmax);
		ki.resize(locals[l].ki.size());
		for (size_t i=0; i<locals[l].ki.size(); ++i){
			ki[i] = static_cast<double>(locals[l].ki[i]);
		}
		h0 = static_cast<double>(locals[l].h[0]);
		h1 = static_cast<double>(locals[l].h[1]);
		h2 = static_cast<double>(locals[l].h[2]);
		h3 = static_cast<double>(locals[l].h[3]);
		w = static_cast<double>(locals[l].w);
		lh0 = static_cast<double>(locals[l].lh0);
		l1p_w = static_cast<double>(locals[l].l1p_w);
		log_scale = static_cast<double>(locals[l].log_scale.log_integrand);
		ymax = static_cast<double>(locals[l].ymax);
		norm = static_cast<double>(locals[l].norm);
	}

	void get_log_pdf_bli_samples(std::vector<std::vector<std::pair<double,double>>>& samples) const
	{
		if (log_pdf_interp_low)
			log_pdf_interp_low->get_samples(samples);
		if (log_pdf_interp_high){
			std::vector<std::vector<std::pair<double,double>>> samples2;
			log_pdf_interp_high->get_samples(samples2);
			samples.insert(samples.end(), samples2.cbegin(),
			               samples2.cend());
		}
	}


	/*
	 * Debugging facilities.
	 */
	void get_C(real a, size_t l,
	           double& C0, double& C1, double& C2, double& C3) const
	{
		if (l >= locals.size())
			throw std::runtime_error("Index 'l' out of bounds.");
		posterior::C1_t c1(a, locals[l]);
		posterior::C2_t c2(a, locals[l]);
		posterior::C3_t c3(a, locals[l]);
		// Save the values:
		C0 = 1.0;
		C1 = static_cast<double>(c1.deriv0);
		C2 = static_cast<double>(c2.deriv0);
		C3 = static_cast<double>(c3.deriv0);
	}

	/*
	 * Serialization:
	 */
//	template<typename istream>
//	Posterior(istream& in) : locals(0), weights(0)

	template<typename ostream>
	void write(ostream& out) const {
		size_t L_n = locals.size();
		out.write(&L_n, sizeof(size_t));
		for (size_t i=0; i<locals.size(); ++i){
			locals[i].write(out);
		}
		for (real w : weights){
			out.write(&w, sizeof(real));
		}
		out.write(&p, sizeof(double));
		out.write(&s, sizeof(double));
		out.write(&n, sizeof(double));
		out.write(&v, sizeof(double));
		out.write(&Qmax, sizeof(real));
		out.write(&norm, sizeof(real));
		out.write(&rtol, sizeof(double));
	}

private:

	std::vector<posterior::LocalsAndNorm<real>> locals;
	/*
	 * The weights contain a lot of stuff:
	 *  - user weights
	 *  - log scale difference between locals
	 * To obtain a properly normalized PDF, the individual results of
	 * `outer_integrand` simply need to be multiplied by the weights,
	 * divided by norm and summed.
	 */
	std::vector<real> weights;
	double p;
	double s;
	double n;
	double v;
	real Qmax;
	real norm;
	double rtol;
	size_t bli_max_splits;
	uint8_t bli_max_refinements;
	real tmin;
	real tmax;

	constexpr static double transition = 0.9;

	mutable std::optional<rn::PiecewiseBarycentricLagrangeInterpolator<real>> log_pdf_interp_low;
	mutable std::optional<rn::PiecewiseBarycentricLagrangeInterpolator<real>> log_pdf_interp_high;
	mutable std::optional<rn::SimpsonAdaptiveQuadrature<real, real>> saq;

	Posterior(std::vector<posterior::LocalsAndNorm<real>>&& locals,
	          std::vector<real>&& weights, double p, double s, double n,
	          double v, double amin, double rtol, real Qmax, real norm,
	          size_t bli_max_splits, uint8_t bli_max_refinements)
	   : locals(std::move(locals)), weights(std::move(weights)), p(p), s(s),
	     n(n), v(v), Qmax(Qmax), norm(norm), rtol(rtol),
	     bli_max_splits(bli_max_splits), bli_max_refinements(bli_max_refinements),
	     tmin(rm::log(std::numeric_limits<double>::epsilon() * Qmax)),
	     tmax(rm::log((1.0 - transition)*Qmax))
	{
		if (pdf_algorithm == BARYCENTRIC_LAGRANGE){
			init_log_pdf_bli(locals, weights, Qmax, norm,
			                 bli_max_splits, bli_max_refinements,
			                 rtol);
		} else if (pdf_algorithm == ADAPTIVE_SIMPSON){
			init_simpson_quadrature();
		}
	}

	static std::vector<posterior::LocalsAndNorm<real>>
	init_locals(const std::vector<weighted_sample_t>& weighted_samples,
	            const double p, const double s, const double n, const double v,
	            const double amin, const double rtol)
	{
		const size_t N = weighted_samples.size();

		std::vector<posterior::LocalsAndNorm<real>> locals(N);

		/* Perform parallel computation only if more than one set of samples: */
		const bool parallel = N > 1;

		if (parallel){
			std::exception_ptr error;

			#pragma omp parallel for
			for (size_t i=0; i<N; ++i)
			{
				if (error)
					continue;

				const weighted_sample_t& ws = weighted_samples[i];
				std::exception_ptr local_error;
				if (std::isnan(ws.w) || ws.w <= 0){
					local_error = std::make_exception_ptr(
						std::runtime_error("NaN, zero, or negative weight found"));
				} else {
					try {
						locals[i] = posterior::LocalsAndNorm<real>(ws.sample, p, s,
																n, v, amin,
																rtol);
					} catch (...) {
						local_error = std::current_exception();
					}
				}
				if (local_error){
					#pragma omp critical
					{
						error = local_error;
					}
				}
			}

			/* Check for any occurred errors: */
			if (error){
				std::string msg = "An error occurred while computing the "
								"posterior attributes: ";
				try {
					std::rethrow_exception(error);
				} catch (const std::runtime_error& e) {
					msg += e.what();
				}
				throw std::runtime_error(msg);
			}
		} else if (N == 1){
			const weighted_sample_t& ws = weighted_samples[0];
			if (std::isnan(ws.w) || ws.w <= 0)
				throw std::runtime_error("NaN, zero, or negative weight found");

			locals[0] = posterior::LocalsAndNorm<real>(ws.sample, p, s, n, v, amin, rtol);
		} else {
			throw std::runtime_error("No samples given.");
		}
		return locals;
	}

	static std::vector<real>
	init_weights(const std::vector<weighted_sample_t>& weighted_samples,
	             const std::vector<posterior::LocalsAndNorm<real>>& locals)
	{
		/* The `outer_integrand` function uses the `log_scale` parameter
		 * to rescale the integrand into a dynamic range that can be handled
		 * by the integrator.
		 * When considering only one sample, that is, one Locals instance,
		 * this is not a problem since the normalization will cancel out any
		 * scale parameter. However, if we want to compare the likelihood
		 * across samples, having an equal `log_scale` becomes important.
		 * Since `log_scale` varies over the elements of `locals`,
		 * we need to readjust.
		 */
		real log_scale = -std::numeric_limits<real>::infinity();
		for (const posterior::LocalsAndNorm<real>& l : locals)
			if (l.log_scale.log_integrand > log_scale)
				log_scale = l.log_scale.log_integrand;

		/*
		 * Compute the weights:
		 */
		std::vector<real> weights;
		for (size_t i=0; i<locals.size(); ++i)
		{
			/* Get the user-provided weight: */
			real wi = weighted_samples[i].w;
			if (rm::isnan(wi) || wi <= 0)
				continue;

			/* Now adjust to the global log scale: */
			wi *= rm::exp(locals[i].log_scale.log_integrand - log_scale);

			weights.push_back(wi);
		}

		/* This should never actually be the case and indicates that
		 * some double arithmetic condition has not been taken into account.
		 */
		if (weights.size() != locals.size())
			throw std::runtime_error("Something unusual happened.");

		return weights;
	}

	static real
	compute_norm(const std::vector<posterior::LocalsAndNorm<real>>& locals,
	             const std::vector<real>& weights)
	{
		/* Now we can properly normalize: */
		auto summand =
		[&](size_t i) -> real {
			return locals[i].norm * weights[i];
		};
		return rn::kahan_sum(locals.size(), summand);
	}

	static real
	global_Qmax(const std::vector<posterior::LocalsAndNorm<real>>& locals)
	{
		real Qmax = 0.0;
		for (const posterior::LocalsAndNorm<real>& l : locals){
			if (l.Qmax > Qmax)
				Qmax = l.Qmax;
		}
		return Qmax;
	}

	static rn::PiecewiseBarycentricLagrangeInterpolator<real>
	init_pdf_bli(const std::vector<posterior::LocalsAndNorm<real>>& locals,
	             const std::vector<real>& weights,
	             typename posterior::arg<const real>::type Qmax,
	             typename posterior::arg<const real>::type norm,
	             size_t max_splits, uint8_t max_refinements,
	             double rtol
	             )
	{
		auto pdf =
		  [&locals, &weights, Qmax, norm](const rn::PointInInterval<real>& x) -> real
		{
			return pdf_single_explicit<pdf_t>(x, locals, weights, Qmax, norm);
		};
		const real tol_rel = rtol;
		const real tol_abs = rtol / Qmax;
		return rn::PiecewiseBarycentricLagrangeInterpolator<real>(
		           pdf, 0.0, Qmax, tol_rel, tol_abs, 0.0,
		           std::numeric_limits<real>::infinity(),
		           max_splits, max_refinements
		);
	}


	void
	init_log_pdf_bli(const std::vector<posterior::LocalsAndNorm<real>>& locals,
	                 const std::vector<real>& weights,
	                 typename posterior::arg<const real>::type Qmax,
	                 typename posterior::arg<const real>::type norm,
	                 size_t max_splits, uint8_t max_refinements,
	                 double rtol
	)
	{
		/* The lower interpolant: */
		auto log_pdf =
		  [&locals, &weights, Qmax, norm](const rn::PointInInterval<real>& x) -> real
		{
			return pdf_single_explicit<log_pdf_t>(x, locals, weights, Qmax, norm);
		};
		const real tol_rel = rtol;
		const real tol_abs = rtol / Qmax;
		log_pdf_interp_low = rn::PiecewiseBarycentricLagrangeInterpolator<real>(
		           log_pdf, 0.0, transition * Qmax, tol_rel, tol_abs,
		           -std::numeric_limits<real>::infinity(),
		           std::numeric_limits<real>::infinity(),
		           max_splits, max_refinements
		);

		/* The upper interpolant: */
		auto log_pdf_high =
		  [&locals, &weights, Qmax, norm](const rn::PointInInterval<real>& t_back) -> real
		{
			/* Flip x back: */
			real x_back = rm::exp(t_back.val);
			rn::PointInInterval<real> x(
			    Qmax-x_back,
			    Qmax-x_back,
			    x_back
			);
			real lpdf = pdf_single_explicit<log_pdf_t>(x, locals, weights, Qmax, norm);
			return lpdf;
		};
		log_pdf_interp_high = rn::PiecewiseBarycentricLagrangeInterpolator<real>(
		           log_pdf_high,
		           tmin, tmax, tol_rel, tol_abs,
		           -std::numeric_limits<real>::infinity(),
		           std::numeric_limits<real>::infinity(),
		           max_splits, max_refinements
		);
	}


	void init_simpson_quadrature() const {
		if (pdf_algorithm == pdf_algorithm_t::BARYCENTRIC_LAGRANGE){
			if (!log_pdf_interp_low || !log_pdf_interp_high)
				throw std::runtime_error("Unexpectedly, `pdf_interp` is not initialized.");
			auto pdf =
			[this](real x) -> real
			{
				rn::PointInInterval<real> x_pii(x, x, Qmax-x);
				return pdf_single<pdf_algorithm_t::BARYCENTRIC_LAGRANGE>(x_pii);
			};
			saq.emplace(pdf, static_cast<real>(0.0), Qmax);
		} else {
			auto pdf =
			[this](real x) -> real
			{
				rn::PointInInterval<real> x_pii(x, x, Qmax-x);
				return pdf_single_explicit<pdf_t>(x_pii, locals, weights, Qmax, norm);
			};
			saq.emplace(pdf, static_cast<real>(0.0), Qmax);
		}
	}

	struct pdf_t {
		static real evaluate(
		    typename posterior::arg<const real>::type zi,
		    const posterior::LocalsAndNorm<real>& l,
		    typename posterior::arg<const real>::type wi,
		    typename posterior::arg<const real>::type norm
		)
		{
			return posterior::outer_integrand<real>(
			            zi, l,
			            l.log_scale.log_integrand
			) * wi / (l.Qmax * norm);
		}

		static real evaluate_large_z(
		    typename posterior::arg<const real>::type yi,
		    const posterior::LocalsAndNorm<real>& l,
		    typename posterior::arg<const real>::type wi,
		    typename posterior::arg<const real>::type norm
		)
		{
			return posterior::a_integral_large_z<false,real>(
			            yi,
			            norm,
			            l.log_scale.log_integrand,
			            l
			) * wi / (l.Qmax * norm);
		}

		static real sum(const std::vector<real>& val)
		{
			rn::KahanAdder<real> res;
			for (size_t j=0; j<val.size(); ++j){
				res += val[j];
			}
			return res;
		}

		static constexpr double zero = 0.0;
	};

	struct log_pdf_t {
		static real evaluate(
		    typename posterior::arg<const real>::type zi,
		    const posterior::LocalsAndNorm<real>& l,
		    typename posterior::arg<const real>::type wi,
		    typename posterior::arg<const real>::type norm
		)
		{
			return posterior::log_outer_integrand<real>(
			            zi, l,
			            l.log_scale.log_integrand
			) + rm::log(wi / (l.Qmax * norm));
		}

		static real evaluate_large_z(
		    typename posterior::arg<const real>::type yi,
		    const posterior::LocalsAndNorm<real>& l,
		    typename posterior::arg<const real>::type wi,
		    typename posterior::arg<const real>::type norm
		)
		{
			return posterior::log_a_integral_large_z<false,real>(
			            yi,
			            norm,
			            l.log_scale.log_integrand,
			            l
			) + rm::log(wi / (l.Qmax * norm));
		}

		static real sum(const std::vector<real>& val)
		{
			/* Find the maximum element: */
			real maxval = *std::max_element(val.cbegin(),
			                                val.cend());

			/* Early exit if the sum is infinite: */
			if (rm::isinf(maxval)){
				return maxval;
			}
			rn::KahanAdder<real> res;
			for (size_t j=0; j<val.size(); ++j){
				real vj = rm::exp(val[j] - maxval);
				res += vj;
			}

			/* Return: */
			return maxval + rm::log(static_cast<real>(res));
		}

		static constexpr double zero = -std::numeric_limits<double>::infinity();
	};


	template<typename T>
	static real pdf_single_explicit_j(const rn::PointInInterval<real>& x,
	                const std::vector<posterior::LocalsAndNorm<real>>& locals,
	                const std::vector<real>& weights,
	                typename posterior::arg<const real>::type Qmax,
	                typename posterior::arg<const real>::type norm,
	                size_t j)
	{
		const real Qmax_j = locals[j].Qmax;
		const real zi = static_cast<real>(x) / Qmax_j;
		if (rm::isnan(zi))
			return std::nan("");
		if (zi > 1.0 || zi < 0.0)
			return T::zero;
		real sj;
		if (zi <= locals[j].ztrans){
			sj = T::evaluate(zi, locals[j], weights[j], norm);
			if (rm::isnan(sj)){
				std::string msg("Found NaN at sub-PDF evaluation (zi=");
				msg += std::to_string(static_cast<long double>(zi));
				msg += ")";
				throw std::runtime_error(msg);
			}
		} else {
			const real yi = (Qmax_j == Qmax)
				? static_cast<real>(x.from_back / Qmax)
				: static_cast<real>(1.0 - zi);

			sj = T::evaluate_large_z(yi, locals[j], weights[j], norm);
			if (rm::isnan(sj)){
				std::string msg("Found NaN at sub-PDF evaluation (zi=");
				msg += std::to_string(static_cast<long double>(zi));
				msg += " - large z)";
				msg += "\nQmax_j: ";
				msg += std::to_string(static_cast<long double>(Qmax_j));
				msg += "\nnorm:   ";
				msg += std::to_string(static_cast<long double>(norm));
				msg += "\nw[j]:   ";
				msg += std::to_string(static_cast<long double>(weights[j]));
				throw std::runtime_error(msg);
			}
		}
		return sj;
	}

	template<typename T, bool serial = false, size_t parallel_min_locals = 2>
	static real pdf_single_explicit(const rn::PointInInterval<real>& x,
	                const std::vector<posterior::LocalsAndNorm<real>>& locals,
	                const std::vector<real>& weights,
	                typename posterior::arg<const real>::type Qmax,
	                typename posterior::arg<const real>::type norm)
	{
		/*
		 * Evaluate the PDF at a single point.
		 */
		real res;
		std::vector<real> resv(locals.size());

		const bool parallel = (!serial) && (locals.size() >= parallel_min_locals);

		if (parallel){
			/*
			 * 1. Parallel Implementation.
			 */
			bool isnan = false;
			std::exception_ptr error;

			#pragma omp parallel for
			for (size_t j=0; j<locals.size(); ++j){
				/* Continue if the result is known Nan: */
				if (isnan)
					continue;

				/* Handle exception capture and propagation across the
				 * OMP border: */
				try {
					resv[j] = pdf_single_explicit_j<T>(
					                x, locals, weights, Qmax,
					                norm, j
					);
				} catch (...) {
					#pragma omp critical
					{
						error = std::current_exception();
					}
				}

				/* Result computed but might be NaN: */
				if (rm::isnan(resv[j]))
					isnan = true;
			}
			if (isnan)
				return std::nan("");

			/* Sum in single thread: */
			res = T::sum(resv);

		} else {
			/*
			 * 2. Serial Implementation.
			 */
			for (size_t j=0; j<locals.size(); ++j){
				resv[j] = pdf_single_explicit_j<T>(
				                x, locals, weights, Qmax,
				                norm, j
				);

				/* Result computed but might be NaN: */
				if (rm::isnan(resv[j]))
					return std::nan("");

			}

			/* Sum: */
			res = T::sum(resv);
		}

		real resr(res);
		if (rm::isnan(resr))
			throw std::runtime_error("Found NaN PDF evaluation.");
		return res;
	}

	/*
	 * Differnt specializations of the PDF evaluated on a single value.
	 * The specializations iterate the backends available in `pdf_algorithm_t`:
	 */
	template<pdf_algorithm_t pa = pdf_algorithm>
	real pdf_single(const std::enable_if_t<pa==BARYCENTRIC_LAGRANGE,
	                                       rn::PointInInterval<real>>& x) const
	{
		/* First, check whether the point is out of
		 * the interpolation domain [0, Qmax-epsilon]
		 * If x == Qmax, we can also safely return 0.
		 */
		if (x.val < 0)
			return 0.0;
		else if (x.from_back <= std::numeric_limits<double>::epsilon())
			return 0.0;

		/* Check for xmin in the logarithmic coordinates
		 * t that will be used in the tail interpolator:
		 */
		real t = rm::log(x.from_back);
		if (t <= tmin)
			return 0.0;

		/* The point is inside the interpolation domain.
		 * Now decide which interpolator to use:
		 */
		if (t >= tmax)
			/* The majority of points will use this
			 * bulk interpolator in [0, 0.9*Qmax] */
			return rm::exp((*log_pdf_interp_low)(x));

		/* From the back. Use the variable `t`, the
		 * logarithm of the distance from the back.
		 * (t € [tmin, tmax])
		 * This transformation leads to a linear
		 * shape of the posterior as it converges
		 * to zero at x==Qmax.
		 * This shape can be easily handled by the barycentric
		 * Lagrange interpolator.
		 */
		rn::PointInInterval<real> tii(
			t,
			t - tmin,
			rm::log((1.0 - transition) * Qmax) - t
		);
		return rm::exp((*log_pdf_interp_high)(tii));
	}

	template<pdf_algorithm_t pa = pdf_algorithm>
	real pdf_single(const std::enable_if_t<pa==ADAPTIVE_SIMPSON,
	                                       rn::PointInInterval<real>>& x) const
	{
		if (static_cast<real>(x) < 0)
			return 0.0;
		else if (static_cast<real>(x) > Qmax)
			return 0.0;
		return saq->density(x);
	}

	template<pdf_algorithm_t pa = pdf_algorithm>
	real pdf_single(const std::enable_if_t<pa==EXPLICIT,
	                                       rn::PointInInterval<real>>& x) const
	{
		if (static_cast<real>(x) < 0)
			return 0.0;
		else if (static_cast<real>(x) > Qmax)
			return 0.0;
		return pdf_single_explicit<pdf_t>(x, locals, weights, Qmax, norm);
	}

};

} // namespace anomaly
} // namespace reheatfunq

#endif