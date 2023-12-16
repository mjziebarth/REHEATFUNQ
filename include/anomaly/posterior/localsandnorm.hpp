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

	LocalsAndLogScale(const std::vector<qc_t>& qc, arg<const real>::type p,
	                  arg<const real>::type s, arg<const real>::type n,
	                  arg<const real>::type v, arg<const real>::type amin,
	                  double dest_tol)
	   : Locals<real>(qc, p, s, n, v, amin, dest_tol),
	     log_scale(log_integrand_max(*this)),
	     ymax(y_taylor_transition(*this)),
	     ztrans(1.0 - ymax)
	{}

	LocalsAndLogScale(arg<const real>::type lp, arg<const real>::type ls,
	                  arg<const real>::type n, arg<const real>::type v,
	                  arg<const real>::type amin, arg<const real>::type Qmax,
	                  std::vector<real>&& ki, const std::array<real,4>& h,
	                  arg<const real>::type w, arg<const real>::type lh0,
	                  arg<const real>::type l1p_w, arg<const real>::type log_scale_a,
	                  arg<const real>::type log_scale_z,
	                  arg<const real>::type log_scale_log_integrand,
	                  arg<const real>::type ymax, arg<const real>::type ztrans)
	   : Locals<real>(lp, ls, n, v, amin, Qmax, std::move(ki), h, w, lh0, l1p_w),
	     log_scale({.a=log_scale_a, .z=log_scale_z,
	                .log_integrand=log_scale_log_integrand}),
	     ymax(ymax), ztrans(ztrans)
	{}

	template<typename istream>
	LocalsAndLogScale(istream& in) : Locals<real>(in)
	{
		in.read(&log_scale.a, sizeof(real));
		in.read(&log_scale.z, sizeof(real));
		in.read(&log_scale.log_integrand, sizeof(real));
		in.read(&ymax, sizeof(real));
		in.read(&ztrans, sizeof(real));
	}

	LocalsAndLogScale() {};

	template<typename ostream>
	void write(ostream& out) const {
		/* Write parent class: */
		Locals<real>::write(out);

		/* These attributes:*/
		out.write(&log_scale.a, sizeof(real));
		out.write(&log_scale.z, sizeof(real));
		out.write(&log_scale.log_integrand, sizeof(real));
		out.write(&ymax, sizeof(real));
		out.write(&ztrans, sizeof(real));
		out << log_scale.a;
		out << log_scale.z;
		out << log_scale.log_integrand;
		out << ymax;
		out << ztrans;
	}
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
		std::optional<reheatfunq::numerics::CDFEval<real>> cdf_eval;
	};
	/*
	 * Attributes:
	 */
	integrals_t integrals;
	real norm;

	LocalsAndNorm(const std::vector<qc_t>& qc, arg<const real>::type p,
	              arg<const real>::type s, arg<const real>::type n,
	              arg<const real>::type v, arg<const real>::type amin,
	              double dest_tol)
	   : LocalsAndLogScale<real>(qc, p, s, n, v, amin, dest_tol),
	     integrals(compute_integrals(*this)),
	     norm(integrals.S + integrals.full_taylor_integral)
	{
		if (rm::isinf(norm) || rm::isnan(norm))
			throw std::runtime_error("Could not compute finite norm.");
		if (norm == 0)
			throw std::runtime_error("Computed zero norm.");
	}

	LocalsAndNorm(const std::vector<qc_t>& qc, arg<const real>::type p,
	              arg<const real>::type s, arg<const real>::type n,
	              arg<const real>::type v, arg<const real>::type amin,
	              const double* x, const size_t Nx,
	              double dest_tol)
	   : LocalsAndLogScale<real>(qc, p, s, n, v, amin, dest_tol),
	     integrals(compute_integrals(*this, x, Nx, dest_tol)),
	     norm(integrals.S + integrals.full_taylor_integral)
	{
		if (rm::isinf(norm) || rm::isnan(norm))
			throw std::runtime_error("Could not compute finite norm.");
		if (norm == 0)
			throw std::runtime_error("Computed zero norm.");
	}

	LocalsAndNorm() {};

	template<typename istream>
	LocalsAndNorm(istream& in) : LocalsAndLogScale<real>(in) {
		in.read(&integrals.S, sizeof(real));
		in.read(&integrals.full_taylor_integral, sizeof(real));
		in.read(&norm, sizeof(real));
	}

	template<typename ostream>
	void write(ostream& out) const {
		/* Write the parent class: */
		 LocalsAndLogScale<real>::write(out);

		/* The additions: */
		out.write(&integrals.S, sizeof(real));
		out.write(&integrals.full_taylor_integral, sizeof(real));
		out.write(&norm, sizeof(real));
	}


private:

	static integrals_t compute_integrals(const LocalsAndLogScale<real>& L)
	{
		const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

		auto integrand = [&L](real z) -> real
		{
			return outer_integrand(z, L, L.log_scale.log_integrand);
		};


		// 1.1: Double numerical integration in z range [0,1-ymax]:
		real S;
		real error;
		bmq::tanh_sinh<real> integrator;
		typedef bmq::gauss_kronrod<real,31> GK;

		try {
			S = integrator.integrate(integrand, static_cast<real>(0),
			                         L.ztrans, TOL_TANH_SINH, &error);
		} catch (...) {
			/* Backup Gauss-Kronrod: */
			real L1;
			S =  GK::integrate(integrand, static_cast<real>(0),
								L.ztrans, 9, TOL_TANH_SINH,
								&error, &L1);
		}

		/*
		 * Taylor integral from z=1-ymax to z=1
		 */
		real full_taylor_integral(
		        a_integral_large_z<true>(L.ymax, S, L.log_scale.log_integrand, L)
		);

		if (rm::isnan(S))
			throw std::runtime_error("`z`-body integral S is NaN.");
		if (rm::isnan(full_taylor_integral))
			throw std::runtime_error("large-`z`, `full_taylor_integral` is NaN");


		return {.S=S, .full_taylor_integral=full_taylor_integral,
		        .cdf_eval=std::optional<reheatfunq::numerics::CDFEval<real>>()};
	}
};




} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq

#endif
