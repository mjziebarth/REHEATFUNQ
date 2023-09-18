/*
 * Heat flow anomaly analysis posterior numerics: integrand in `z`.
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

#ifndef REHEATFUNQ_ANOMALY_POSTERIRO_INTEGRAND_HPP
#define REHEATFUNQ_ANOMALY_POSTERIRO_INTEGRAND_HPP

/*
 * General includes:
 */
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>


/*
 * REHEATFUNQ includes:
 */
#include <anomaly/posterior/exceptions.hpp>
#include <anomaly/posterior/args.hpp>
#include <anomaly/posterior/locals.hpp>

namespace reheatfunq {
namespace anomaly {
namespace posterior {

namespace rm = reheatfunq::math;
namespace bmq = boost::math::quadrature;

/*
 * Get a scale factor for the integrals by computing an approximate maximum
 * of the integrand of the alpha integral (disregarding the z-integral, i.e.
 * assuming constant z):
 */

template<typename real>
real log_integrand_amax(const typename arg<real>::type l1pwz,
                        const typename arg<real>::type lkiz_sum,
                        const Locals<real>& L)
{
	/* Uses Newton-Raphson to compute the (approximate) maximum of the
	 * integrand (disregarding the second integral) of the normalization
	 * of the posterior.
	 */
	const real C = L.lp - L.v*(L.ls + l1pwz) + lkiz_sum;
	real a = 1.0;
	real f0, f1, da;

	bool success = false;
	for (size_t i=0; i<100; ++i){
		f0 = L.v * rm::digamma(L.v * a) - L.n * rm::digamma(a) + C;
		f1 = L.v * L.v * rm::trigamma(L.v * a) - L.n * rm::trigamma(a);
		da = f0 / f1;
		a -= da;
		a = std::max<real>(a, 1e-8);
		success = rm::abs(da) <= 1e-8 * a;
		if (success)
			break;
	}
	if (!success)
		throw std::runtime_error("Could not determine log_integrand_max.");
	return a;
}


template<typename real>
struct itgmax_t {
	real a;
	real logI;
};


template<typename real>
itgmax_t<real> log_integrand_maximum(const typename arg<real>::type l1pwz,
                                     const typename arg<real>::type lkiz_sum,
                                     const Locals<real>& L)
{
	itgmax_t<real> res;
	res.a = log_integrand_amax(l1pwz, lkiz_sum, L);

	res.logI = rm::lgamma(L.v * res.a) + (res.a - 1.0) * L.lp
	           - L.n * rm::lgamma(res.a) - L.v * res.a * (L.ls + l1pwz)
	           + (res.a - 1) * lkiz_sum;

	return res;
}


/*
 * The innermost integrand of the double integral; the integrand in `a`.
 */
template<bool log_integrand, typename real>
real inner_integrand_template(const typename arg<real>::type a,
                              const typename arg<real>::type l1p_kiz_sum,
                              const typename arg<real>::type l1p_wz,
                              const typename arg<real>::type log_integrand_max,
                              const Locals<real>& L)
{

	auto va = L.v * a;

	/*
	 * Shortcut for small a
	 * With SymPy, we find the following limit for a -> 0:
	 *    loggamma(v*a) - v*n*loggamma(a) --> -inf * sign(n - 1)
	 * Since for finite data sets, n>1 (the number of data points
	 * will be added), we hence have lS -> -inf and thus S -> 0
	 */
	if (a == 0)
		return 0;
	const real lga = rm::lgamma(a);
	if (rm::isinf(lga))
		return 0;
	const real lgva = rm::lgamma(va);
	if (rm::isinf(lgva))
		return 0;


	// Term PROD_i{  (1-k[i] * z) ^ (a-1)  }
	real lS = (a - 1) * l1p_kiz_sum;

	// Term ( 1 / (s_new * (1-w*z))) ^ va
	lS -= va * (L.ls + l1p_wz);

	// Remaining summands:
	lS += lgva + (a - 1.0) * L.lp - L.n * lga - log_integrand_max;

	// Shortcut for debugging purposes:
	if (log_integrand)
		return lS;

	// Compute the result and test for finiteness:
	real result = rm::exp(lS);
	if (rm::isinf(result)){
		throw ScaleError<real>("inner_integrand", lS);
	}

	if (rm::isnan(result)){
		std::string msg("inner_integrand: NaN result at a =");
		msg.append(std::to_string(static_cast<double>(a)));
		msg.append(". lS = ");
		msg.append(std::to_string(static_cast<double>(lS)));
		throw std::runtime_error(msg);
	}

	return result;
}


template<typename real>
real outer_integrand(const typename arg<real>::type z, const Locals<real>& L,
                     const typename arg<real>::type log_scale,
                     const typename arg<real>::type Iref = 0)
{
	const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

	if (rm::isnan(z))
		throw std::runtime_error("`z` is nan!");

	// Set the inner parameters:
	real l1p_kiz_sum = 0.0;
	for (const real& k : L.ki)
		l1p_kiz_sum += rm::log1p(-k * z);
	const real l1p_wz = rm::log1p(-L.w * z);

	/* The non-degenerate case.
	 * First compute the maximum of the a integrand at this given z:
	 */
	const itgmax_t<real> lImax = log_integrand_maximum(l1p_wz, l1p_kiz_sum, L);

	// Integrate:
	auto integrand = [&](const typename arg<real>::type a) -> real {
		return inner_integrand_template<false>(a, l1p_kiz_sum, l1p_wz,
		                                       lImax.logI, L);
	};

	real error, L1, S;
	try {
		/* We wrap the integration into a try-catch to be able to distinguish
		 * the source of the error if any should be thrown.
		 */
		if (lImax.a > L.amin){
			size_t lvl0, lvl1;
			real error1, L11;
			bmq::tanh_sinh<real> int0;
			bmq::exp_sinh<real> int1;
			S = int0.integrate(integrand, L.amin, lImax.a,
			                   TOL_TANH_SINH, &error, &L1, &lvl0)
			  + int1.integrate(integrand, lImax.a,
			                   std::numeric_limits<real>::infinity(),
			                   TOL_TANH_SINH, &error1, &L11, &lvl1);
			L1 += L11;
			error += error1;
		} else {
			size_t levels;
			bmq::tanh_sinh<real> integrator;
			S = integrator.integrate(integrand, L.amin,
			                         std::numeric_limits<real>::infinity(),
			                         TOL_TANH_SINH, &error, &L1, &levels);
		}
	} catch (std::runtime_error& e){
		std::string msg("Runtime error in outer_integrand tanh_sinh "
		                "quadrature.\n Message: '");
		msg += std::string(e.what());
		msg += "'.";
		throw std::runtime_error(msg);
	}

	if (rm::isinf(S)){
		throw ScaleError<real>("outer_integrand", log_scale);
	}

	/* Error checking: */
	constexpr double ltol = std::log(1e-5);
	const real cmp_err
	   = std::max<real>(rm::log(L1), rm::log(Iref) + log_scale
	                                 - lImax.logI);
	if (rm::log(error) > ltol + cmp_err){
		/* Check if this is relevant: */
		throw PrecisionError<real>("outer_integrand", error, L1);
	}

	return S * rm::exp(lImax.logI - log_scale);
}



} // namespace anomaly
} // namespace posterior
} // namespace reheatfunq

#endif