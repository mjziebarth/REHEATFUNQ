/*
 * Heat flow anomaly analysis posterior numerics: locals ('locals.hpp') extended
 *                                                by normalization.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2021-2022 Deutsches GeoForschungsZentrum GFZ,
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
 */

#ifndef REHEATFUNQ_ANOMALY_POSTERIOR_NORMEDLOCALS_HPP
#define REHEATFUNQ_ANOMALY_POSTERIOR_NORMEDLOCALS_HPP

/*
 * General includes:
 */
#include <map>
#include <optional>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

/*
 * REHEATFUNQ includes
 */
#include <anomaly/posterior/locals.hpp>
#include <anomaly/posterior/amax.hpp>
#include <anomaly/posterior/large_z.hpp>
#include <numerics/cdfeval.hpp>

namespace reheatfunq {
namespace anomaly {
namespace posterior {

namespace bmq = boost::math::quadrature;


template<typename real>
class LocalsAndLogScale : public Locals<real>
{
public:
	/*
	 * Attributes:
	 */
	az_t<real> log_scale;
	real ymax;
	real ztrans;

	LocalsAndLogScale(const std::vector<qc_t>& qc, const arg<real>::type p,
	                  const arg<real>::type s, const arg<real>::type n,
	                  const arg<real>::type v, const arg<real>::type amin,
	                  double dest_tol)
	   : Locals<real>(qc, p, s, n, v, amin, dest_tol),
	     log_scale(log_integrand_max(*this)),
	     ymax(y_taylor_transition(*this, static_cast<real>(1e-32))),
	     ztrans(1.0 - ymax)
	{}

	LocalsAndLogScale() {};
};




template<typename real>
class LocalsAndNorm : public LocalsAndLogScale<real>
{
public:
	/*
	 * Types:
	 */
	struct integrals_t {
		real S;
		real full_taylor_integral;
		std::optional<reheatfunq::CDFEval<real>> cdf_eval;
	};
	/*
	 * Attributes:
	 */
//	const real Iref;
	integrals_t integrals;
	real norm;

	LocalsAndNorm(const std::vector<qc_t>& qc, const arg<real>::type p,
	              const arg<real>::type s, const arg<real>::type n,
	              const arg<real>::type v, const arg<real>::type amin,
	              double dest_tol)
	   : LocalsAndLogScale<real>(qc, p, s, n, v, amin, dest_tol),
//	     Iref(outer_integrand(
//	              LocalsAndLogScale<real>::log_scale.z,
//	              *this,
//	              LocalsAndLogScale<real>::log_scale.log_integrand,
//	              0.0)),
	     integrals(compute_integrals(*this)),
	     norm(integrals.S + integrals.full_taylor_integral)
	{}

	LocalsAndNorm(const std::vector<qc_t>& qc, const arg<real>::type p,
	              const arg<real>::type s, const arg<real>::type n,
	              const arg<real>::type v, const arg<real>::type amin,
	              const double* x, const size_t Nx,
	              double dest_tol)
	   : LocalsAndLogScale<real>(qc, p, s, n, v, amin, dest_tol),
//	     Iref(outer_integrand(LocalsAndLogScale<real>::log_scale.z, *this)),
	     integrals(compute_integrals(*this, x, Nx, dest_tol)),
	     norm(integrals.S + integrals.full_taylor_integral)
	{}

	LocalsAndNorm() {};


private:

	static integrals_t compute_integrals(const LocalsAndLogScale<real>& L)
	{
		const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

		auto integrand = [&](real z) -> real
		{
			return outer_integrand(z, L, L.log_scale.log_integrand);
		};

		// 1.1: Double numerical integration in z range [0,1-ymax]:
		real S;
		real error;
		bmq::tanh_sinh<real> integrator;
		try {
			S = integrator.integrate(integrand, static_cast<real>(0),
			                         L.ztrans, TOL_TANH_SINH, &error);
		} catch (...) {
			/* Backup Gauss-Kronrod: */
			typedef bmq::gauss_kronrod<real,31> GK;
			real L1;
			S =  GK::integrate(integrand, static_cast<real>(0),
			                   L.ztrans, 9, TOL_TANH_SINH,
			                   &error, &L1);
		}
		// TODO: Check result!

		/*
		 * Taylor integral from z=1-ymax to z=1
		 */
		real full_taylor_integral(
		        a_integral_large_z<true>(L.ymax, S, L.log_scale.log_integrand, L)
		);


		return {.S=S, .full_taylor_integral=full_taylor_integral,
		        .cdf_eval=std::optional<CDFEval<real>>()};
	}

	static real compute_integrals(const LocalsAndLogScale<real>& L,
	                              const double* x, size_t Nx, double dest_tol)
	{
		const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

		std::map<real,real> cache;

		auto integrand = [&](real z) -> real
		{
			auto it = cache.lower_bound(z);
			if (it == cache.end() || it->first != z){
				real val = outer_integrand(z, L);
				it = cache.insert(it, std::make_pair(z, val));
			}
			return it->second;
		};

		// 1.1: Double numerical integration in z range [0,1-ymax]:
		/* This is called for the CDF: */
		std::vector<real> z;
		z.reserve(Nx);
		for (size_t i=0; i<Nx; ++i){
			real zi = static_cast<real>(x[i]) / L.Qmax;
			if (zi <= L.ztrans)
				z.push_back(zi);
		}
		CDFEval<real> cdf_eval(z, integrand, 0.0, L.ztrans,
		                       0.0, dest_tol);
		real S = cdf_eval.norm();

		/*
		 * Taylor integral from z=1-ymax to z=1
		 */
		real full_taylor_integral(
		        a_integral_large_z<true>(L.ymax, S, L.log_scale.log_integrand, L)
		);


		return {.S=S, .full_taylor_integral=full_taylor_integral,
		        .cdf_eval=cdf_eval};
	}
};




} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq

#endif
