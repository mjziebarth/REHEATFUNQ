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
#define BOOST_ENABLE_ASSERT_HANDLER // Make sure the asserts do not abort
#include <vector>
#include <algorithm>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/tools/roots.hpp>

#include <chrono>
#include <thread>
#include <string>
#include <sstream>



/*
 * REHEATFUNQ includes
 */
#include <anomaly/posterior/localsandnorm.hpp>
#include <numerics/kahan.hpp>
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

template<typename real>
class Posterior {
public:

	Posterior(const std::vector<weighted_sample_t>& weighted_samples,
	          double p, double s, double n, double v, double amin,
	          double dest_tol)
	   : locals(init_locals(weighted_samples, p, s, n, v, amin, dest_tol)),
	     weights(init_weights(weighted_samples, locals)),
	     p(p), s(s), n(n), v(v), Qmax(global_Qmax(locals)),
	     norm(compute_norm(locals, weights)), dest_tol(dest_tol),
	     pdf_interp(init_pdf_bli(locals, weights, Qmax, norm))
	{
	}

	double get_Qmax() const {
		return Qmax;
	}

	template<bool use_bli=true>
	void pdf(std::vector<double>& PH) const
	{
		const size_t Nx = PH.size();
		std::vector<double>& result = PH;

		std::optional<std::exception> except;
		#pragma omp parallel for schedule(guided)
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
				result[i] = pdf_single<use_bli>(PH[i]);
			} catch (const std::exception& e) {
				except = e;
			}
		}

		if (except)
			throw except;
	}

	template<bool use_bli=true>
	void cdf(std::vector<double>& PH, bool parallel=true,
	         bool adaptive=false) const
	{
		const size_t Nx = PH.size();

		if (use_bli){
			if (!cdf_interp)
				cdf_interp.emplace(init_cdf_bli(locals, weights, Qmax, norm));

			/*
			 * OMP-compatible exception propagation code:
			 */
			std::exception_ptr error;

			#pragma omp parallel if(parallel)
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
						PH[i] = (*cdf_interp)(PH[i]);
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

		} else {
			std::vector<real> result(PH.size());

			/*
			 * Compute the CDF via successive integration over the intervals.
			 * The approach varies depending on whether the input points are sorted
			 * or not.
			 */
			if (std::is_sorted(PH.cbegin(), PH.cend()))
			{
				/* Already sorted. */
				cdf_sorted<false>(PH, result, adaptive, parallel);

				for (size_t i=0; i<Nx; ++i)
					PH[i] = result[i];

			} else {
				/*
				 * Create the ordering for successive integration:
				 */
				struct order_t {
					double x;
					size_t i;
				};
				std::vector<order_t> order(Nx);
				for (size_t i=0; i<Nx; ++i){
					order[i].i = i;
					order[i].x = PH[i];
				}
				std::sort(order.begin(), order.end(),
				          [&PH](order_t o0, order_t o1) -> bool
				          {
				              return o0.x < o1.x;
				          });
				for (size_t i=0; i<Nx; ++i){
					PH[i] = order[i].x;
				}

				/*
				 * Call the algorithm on the ordered vector:
				 */
				cdf_sorted<false>(PH, result, adaptive, parallel);

				/*
				 * Transfer the results:
				 */
				for (size_t i=0; i<Nx; ++i)
					PH[order[i].i] = result[i];
			}
		}
	}


	template<typename real_in=double, bool use_bli=true>
	void tail(std::vector<real_in>& PH, bool parallel=true,
	          bool adaptive=false) const
	{
		const size_t Nx = PH.size();

		if (use_bli){
			if (!tail_interp)
				tail_interp.emplace(init_tail_bli(locals, weights, Qmax, norm));

			/*
			 * OMP-compatible exception propagation code:
			 */
			std::exception_ptr error;

			#pragma omp parallel if(parallel)
			for (size_t i=0; i<Nx; ++i){
				if (error)
					continue;
				std::exception_ptr local_error;
				try {
					/*
					 * The actual numerical code:
					 */
					PH[i] = (*tail_interp)(PH[i]);
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

		} else {
			std::vector<real> result(PH.size());

			/*
			 * Compute the CDF via successive integration over the intervals.
			 * The approach varies depending on whether the input points are sorted
			 * or not.
			 */
			if (std::is_sorted(PH.cbegin(), PH.cend()))
			{
				/* Already sorted. */
				std::cout << "sorted.\n" << std::flush;
				cdf_sorted<true>(PH, result, adaptive, parallel);

				for (size_t i=0; i<Nx; ++i)
					PH[i] = result[i];

			} else {
				/*
				 * Create the ordering for successive integration:
				 */
				std::cout << "not sorted.\n" << std::flush;
				struct order_t {
					double x;
					size_t i;
				};
				std::vector<order_t> order(Nx);
				for (size_t i=0; i<Nx; ++i){
					order[i].i = i;
					order[i].x = PH[i];
				}
				std::sort(order.begin(), order.end(),
				          [&PH](order_t o0, order_t o1) -> bool
				          {
				              return o0.x < o1.x;
				          });
				for (size_t i=0; i<Nx; ++i){
					PH[i] = order[i].x;
				}

				/*
				 * Call the algorithm on the ordered vector:
				 */
				cdf_sorted<true>(PH, result, adaptive, parallel);

				/*
				 * Transfer the results:
				 */
				for (size_t i=0; i<Nx; ++i)
					PH[order[i].i] = result[i];
			}
		}
	}

	template<bool use_bli=true>
	void tail_quantiles(std::vector<double>& quantiles, size_t n_chebyshev=100,
	                    bool parallel=true, bool adaptive=false) const
	{
		if (use_bli){
			if (!tail_interp)
				tail_interp.emplace(init_tail_bli(locals, weights, Qmax, norm));

			/*
			 * OMP-compatible exception propagation code:
			 */
			std::exception_ptr error;

			#pragma omp parallel if(parallel)
			for (size_t i=0; i<quantiles.size(); ++i){
				if (error)
					continue;

				std::exception_ptr local_error;
				if (quantiles[i] == 0.0)
					quantiles[i] = Qmax;
				else if (quantiles[i] == 1.0)
					quantiles[i] = 0.0;
				else if (quantiles[i] > 0.0 && quantiles[i] < 1.0){
					/* The typical case. Use TOMS 748 on a quantile
					 * function to find the quantile.
					 */
					const real qi = quantiles[i];
					auto quantile_function =
					[this,qi](real PH) -> real
					{
						return (*tail_interp)(PH) - qi;
					};
					std::uintmax_t max_iter(100);
					bmt::eps_tolerance<real>
					   eps_tol(std::numeric_limits<real>::digits - 2);
					try {
						std::pair<real,real> bracket
						   = bmt::toms748_solve(quantile_function,
						                    static_cast<real>(0.0), Qmax,
						                    static_cast<real>(1.0-quantiles[i]),
						                    static_cast<real>(-quantiles[i]),
						                    eps_tol, max_iter);
						quantiles[i] = 0.5*(bracket.first + bracket.second);
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
		} else {
			tail_quantiles_old(quantiles, n_chebyshev, parallel, adaptive);
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
		   new_locals(init_locals(weighted_samples, p, s, n, v, locals[0].amin,
		                         dest_tol));
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
		                 n, v, locals[0].amin, dest_tol, res_Qmax, res_norm);
	}


	/*
	 * Validation against the previous implementation; not for general use.
	 */
	bool validate(const std::vector<std::vector<posterior::qc_t>>& qc_set,
	              double p0, double s0, double n0, double v0,
	              double dest_tol) const;

	void get_locals(size_t l, double& lp, double& ls, double& n, double& v,
	                double& amin, double& Qmax, std::vector<double>& ki,
	                double& h0, double& h1, double& h2, double& h3,
	                double& w, double& lh0, double& l1p_w, double& log_scale,
					double& ymax, double& norm) const
	{
		if (l >= locals.size())
			throw std::runtime_error("Index 'l' out of bounds.");
		lp = locals[l].lp;
		ls = locals[l].ls;
		n = locals[l].n;
		v = locals[l].v;
		amin = locals[l].amin;
		Qmax = locals[l].Qmax;
		ki.resize(locals[l].ki.size());
		for (size_t i=0; i<locals[l].ki.size(); ++i){
			ki[i] = locals[l].ki[i];
		}
		h0 = locals[l].h[0];
		h1 = locals[l].h[1];
		h2 = locals[l].h[2];
		h3 = locals[l].h[3];
		w = locals[l].w;
		lh0 = locals[l].lh0;
		l1p_w = locals[l].l1p_w;
		log_scale = locals[l].log_scale.log_integrand;
		ymax = locals[l].ymax;
		norm = locals[l].norm;
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
		C1 = c1.deriv0;
		C2 = c2.deriv0;
		C3 = c3.deriv0;
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
	double dest_tol;

	rn::PiecewiseBarycentricLagrangeInterpolator<real> pdf_interp;
	mutable std::optional<rn::PiecewiseBarycentricLagrangeInterpolator<real>>
	    cdf_interp;
	mutable std::optional<rn::PiecewiseBarycentricLagrangeInterpolator<real>>
	    tail_interp;

	Posterior(std::vector<posterior::LocalsAndNorm<real>>&& locals,
	          std::vector<real>&& weights, double p, double s, double n,
	          double v, double amin, double dest_tol, real Qmax, real norm)
	   : locals(std::move(locals)), weights(std::move(weights)), p(p), s(s),
	     n(n), v(v), Qmax(Qmax), norm(norm),
	     pdf_interp(init_pdf_bli(locals, weights, Qmax, norm))
	{
	}

	static std::vector<posterior::LocalsAndNorm<real>>
	init_locals(const std::vector<weighted_sample_t>& weighted_samples,
	            const double p, const double s, const double n, const double v,
	            const double amin, const double dest_tol)
	{
		const size_t N = weighted_samples.size();

		std::exception_ptr error;
		std::vector<posterior::LocalsAndNorm<real>> locals(N);

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
					                                           dest_tol);
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
		if (locals.empty())
			throw std::runtime_error("No sample with positive weights "
			                         "provided.");
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
			if (std::isnan(wi) || wi <= 0)
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
	             const posterior::arg<real>::type Qmax,
	             const posterior::arg<real>::type norm)
	{
		auto pdf =
		  [&locals, &weights, &norm](const posterior::arg<real>::type x) -> real
		{
			return pdf_single_explicit(x, locals, weights, norm);
		};
		return rn::PiecewiseBarycentricLagrangeInterpolator<real>(pdf, 0.0,
		                                                          Qmax);
	}


	static rn::PiecewiseBarycentricLagrangeInterpolator<real>
	init_cdf_bli(const std::vector<posterior::LocalsAndNorm<real>>& locals,
	             const std::vector<real>& weights,
	             const posterior::arg<real>::type Qmax,
	             const posterior::arg<real>::type norm)
	{
		auto pdf =
		  [&locals, &weights, &norm](const posterior::arg<real>::type x) -> real
		{
			return pdf_single_explicit(x, locals, weights, norm);
		};
		auto cdf = [pdf, Qmax](const posterior::arg<real>::type x) -> real
		{
			if (x <= 0)
				return 0.0;
			else if (x >= Qmax)
				return 1.0;

			/* Integrate: */
			const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();
			bmq::tanh_sinh<real> integrator;
			real error, L1;
			size_t lvl;
			real I = integrator.integrate(pdf, 0.0, x, TOL_TANH_SINH, &error,
			                              &L1, &lvl);
			if (error / L1 > TOL_TANH_SINH)
				throw std::runtime_error("Large error in init_cdf_bli.");
			return std::min<real>(std::max<real>(I, 0.0), 1.0);
		};
		const real tol_rel = std::sqrt(std::numeric_limits<real>::epsilon());
		const real tol_abs = std::numeric_limits<real>::infinity();
		return rn::PiecewiseBarycentricLagrangeInterpolator<real>(cdf, 0.0,
		                                  Qmax, tol_rel, tol_abs, 0.0, 1.0);
	}


	static rn::PiecewiseBarycentricLagrangeInterpolator<real>
	init_tail_bli(const std::vector<posterior::LocalsAndNorm<real>>& locals,
	              const std::vector<real>& weights,
	              const posterior::arg<real>::type Qmax,
	              const posterior::arg<real>::type norm)
	{
		auto pdf =
		  [&locals, &weights, &norm](const posterior::arg<real>::type x) -> real
		{
			return pdf_single_explicit(x, locals, weights, norm);
		};
		auto tail = [pdf, Qmax, &locals](const posterior::arg<real>::type x) -> real
		{
			if (x <= 0)
				return 1.0;
			else if (x >= Qmax)
				return 0.0;

			/* Integrate: */
			const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();
			bmq::tanh_sinh<real> integrator;
			real error, L1;
			size_t lvl;
			real I = integrator.integrate(pdf, x, Qmax, TOL_TANH_SINH, &error,
			                              &L1, &lvl);
			if (error / L1 > TOL_TANH_SINH){
				std::stringstream msg;
				msg << "Large error in init_tail_bli. At x=";
				msg << std::setprecision(20);
				msg << x << ", z=" << x / Qmax;
				msg << ".\nerror =" << error;
				msg << ".\nL1    =" << L1
				    << ".\nlevel =" << lvl << "\n";
				msg << ".\nxmax  =" << locals[0].ztrans * locals[0].Qmax << "\n";
				throw std::runtime_error(msg.str());
			}
			return std::min<real>(std::max<real>(I, 0.0), 1.0);
		};
		const real tol_rel = std::sqrt(std::numeric_limits<real>::epsilon());
		const real tol_abs = std::numeric_limits<real>::infinity();
		return rn::PiecewiseBarycentricLagrangeInterpolator<real>(tail, 0.0,
		                                  Qmax, tol_rel, tol_abs, 0.0, 1.0);
	}



	static real pdf_single_explicit(const posterior::arg<real>::type x,
	                const std::vector<posterior::LocalsAndNorm<real>>& locals,
	                const std::vector<real>& weights,
	                const posterior::arg<real>::type norm)
	{
		/*
		 * Evaluate the PDF at a single point.
		 */
		rn::KahanAdder<real> res;

		for (size_t j=0; j<locals.size(); ++j){
			const real Qmax_j = locals[j].Qmax;
			const real zi = static_cast<real>(x) / Qmax_j;
			if (rm::isnan(zi)){
				return std::nan("");
			}
			if (zi > 1.0 || zi < 0.0)
				continue;
			if (zi <= locals[j].ztrans){
				res += posterior::outer_integrand<real>(zi, locals[j],
				             locals[j].log_scale.log_integrand)
				       * weights[j] / (Qmax_j * norm);
			} else {
				res += posterior::a_integral_large_z<false,real>(1.0 - zi,
				           locals[j].norm, locals[j])
				       * weights[j] / (Qmax_j * norm);
			}
		}
		return res;
	}


	template<bool use_bli>
	real pdf_single(const posterior::arg<real>::type x) const {
		if (use_bli){
			return pdf_single_explicit(x, locals, weights, norm);
		}
		if (x < 0)
			return 0.0;
		else if (x > Qmax)
			return 0.0;
		return pdf_interp(x);
	}


	template<bool tail, typename real_in=double, bool use_bli=true>
	void cdf_sorted(const std::vector<real_in>& x,
	                std::vector<real>& res, bool adaptive,
	                bool parallel) const
	{
		if (x.size() == 0)
			return;
		const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();
		typedef bmq::gauss_kronrod<real,15> GK;
		typedef bmq::gauss<real,7> GL;

		auto integrand = [this](const posterior::arg<real>::type x) -> real {
			return pdf_single<use_bli>(x);
		};

		/* Start from 0.0: */
		if (parallel){
			std::optional<std::runtime_error> rerr;
			#pragma omp parallel for
			for (size_t i=0; i<x.size(); ++i){
				if (rerr)
					continue;
				real_in last_x, next_x;
				if (tail){
					last_x = x[i];
					next_x = (i == x.size()-1) ? Qmax : x[i+1];
				} else {
					last_x = (i == 0) ? 0.0 : x[i-1];
					next_x = x[i];
				}
				real error, L1;
				try {
					if (adaptive)
						res[i] = GK::integrate(integrand, last_x, next_x, 9,
						                       TOL_TANH_SINH, &error, &L1);
					else
						res[i] = GL::integrate(integrand, last_x, next_x, &L1);
				} catch (const std::runtime_error& e){
					#pragma omp critical
					{
						rerr = e;
					}
				}
			}
			if (rerr)
				throw *rerr;
		} else {
			for (size_t i=0; i<x.size(); ++i){
				real_in last_x, next_x;
				if (tail){
					last_x = x[i];
					next_x = (i == x.size()-1) ? Qmax : x[i+1];
				} else {
					last_x = (i == 0) ? 0.0 : x[i-1];
					next_x = x[i];
				}
				real error, L1;
				if (adaptive)
					res[i] = GK::integrate(integrand, last_x, next_x, 9,
					                       TOL_TANH_SINH, &error, &L1);
				else
					res[i] = GL::integrate(integrand, last_x, next_x, &L1);
			}
		}

		if (tail){
			size_t i = x.size() - 1;
			rn::KahanAdder<real> cdfi(res[i]);
			for (; i>0; --i){
				cdfi += res[i];
				res[i] = cdfi;
			}
			cdfi += res[0];
			res[0] = cdfi;
		} else {
			rn::KahanAdder<real> cdfi(res[0]);
			for (size_t i=1; i<x.size(); ++i){
				cdfi += res[i];
				res[i] = cdfi;
			}
		}
	}

	void tail_quantiles_old(std::vector<double>& quantiles,
	                        size_t n_chebyshev, bool parallel,
	                        bool adaptive) const
	{
		if (n_chebyshev <= 1)
			throw std::runtime_error("Need at least 2 Chebyshev points.");

		/* The support for the interpolation: */
		struct xf_t {
			double x;
			real f;
		};
		std::vector<xf_t> support(n_chebyshev, xf_t({0.0, 0.0}));
		{
			/* Prepare the interpolation points: */
			for (size_t i=0; i<n_chebyshev; ++i){
				constexpr long double pi = std::numbers::pi_v<long double>;
				real z = std::cos(i * pi / (n_chebyshev-1));
				support[i].x = std::min(std::max<double>(0.5 * (1.0+z) * Qmax,
				                                         0.0),
				                        (double)Qmax);
			}

			/*
			 * Evaluate the posterior tail.
			 * We fill f with x values - but in reverse order since `x`
			 * is monotonously decreasing.
			 */
			std::vector<real> f(n_chebyshev);
			for (size_t i=0; i<n_chebyshev; ++i){
				f[i] = support[n_chebyshev-i-1].x;
			}
			tail<real>(f, parallel, adaptive);

			/* Transfer to the support vector: */
			for (size_t i=0; i<n_chebyshev; ++i){
				support[i].f = f[n_chebyshev - i - 1];
			}
		}

		/* Now we can interpolate: */
		auto tail_bli = [&support,n_chebyshev](double PH) -> double {
			auto xfit = support.cbegin();
			if (PH == xfit->x)
				return xfit->f;
			rn::KahanAdder<real> nom(0.0);
			rn::KahanAdder<real> denom(0.0);
			real wi = 0.5 / (PH - xfit->x);
			nom += wi * xfit->f;
			denom += wi;
			int sign = -1;
			++xfit;
			for (size_t i=1; i<n_chebyshev-1; ++i){
				if (PH == xfit->x)
					return xfit->f;
				wi = sign * 1.0 / (PH - xfit->x);
				nom += wi * xfit->f;
				denom += wi;
				sign = -sign;
				++xfit;
			}
			if (PH == xfit->x)
				return xfit->f;
			wi = sign * 0.5 / (PH - xfit->x);
			nom += wi * xfit->f;
			denom += wi;
			return std::max<double>(
			          std::min<double>(static_cast<double>(nom)
			                           / static_cast<double>(denom),
			                           1.0),
			          0.0
			);
		};

		/* Now solve for the quantiles: */
		for (size_t i=0; i<quantiles.size(); ++i){
			if (quantiles[i] == 0.0)
				quantiles[i] = Qmax;
			else if (quantiles[i] == 1.0)
				quantiles[i] = 0.0;
			else if (quantiles[i] > 0.0 && quantiles[i] < 1.0){
				/* The typical case. Use TOMS 748 on a quantile
				 * function to find the quantile.
				 */
				const double qi = quantiles[i];
				auto quantile_function =
				[&tail_bli,qi](double PH) -> double
				{
					return tail_bli(PH) - qi;
				};
				std::uintmax_t max_iter(100);
				bmt::eps_tolerance<double>
				   eps_tol(std::numeric_limits<double>::digits - 2);
				std::pair<double,double> bracket
				   = bmt::toms748_solve(quantile_function, 0.0, (double)Qmax,
				                        1.0-quantiles[i], -quantiles[i],
				                        eps_tol, max_iter);
				quantiles[i] = 0.5*(bracket.first + bracket.second);
			} else {
				throw std::runtime_error("Encountered quantile out of bounds "
				                         "[0,1].");
			}
		}
	}


};

} // namespace anomaly
} // namespace reheatfunq

#endif