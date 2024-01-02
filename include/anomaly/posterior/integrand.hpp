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
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/minima.hpp>
#include <iostream>

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
real log_rising_factorial(typename arg<const real>::type a,
                          typename arg<const real>::type log_a,
                          int n)
{
	/*
	 * This implements algorithm 1 of Johansson (2023).
	 *
	 * An upper bound on when to stop the multiplications due to possible overflow: */
	const real max_y = std::numeric_limits<real>::max() / (a+n);
	if (n <= 30){
		real res = log_a;
		/* Keep a running product in y, but do not multiply too often
		 * so as not to overflow: */
		real y = 1.0;
		int mul = 0;
		for (int i=1; i<n; ++i){
			y *= (a+i);
			++mul;
			if (y >= max_y){
				res += rm::log(y);
				y = 1.0;
				mul = 0;
			}
		}
		if (mul > 0)
			res += rm::log(y);
		return res;
	}
	int m = n / 2;
	return log_rising_factorial<real>(a, log_a, m)
	     + log_rising_factorial<real>(a+m, rm::log(a+m), n-m);
}


template<typename real>
real loggamma_v_a__minus__n_loggamma_a__plus__C_a_explicit(
        typename arg<const real>::type a,
        typename arg<const real>::type n,
        typename arg<const real>::type v,
        typename arg<const real>::type C
)
{
	/* Otherwise compute the difference of the log-gamma functions: */
	return rm::lgamma(v * a) - n * rm::lgamma(a) + C * a;
}


template<typename real>
struct log_2_pi {
	const static real value;
};

template<typename real>
real loggamma_v_a__minus__n_loggamma_a__plus__C_a(
        typename arg<const real>::type a,
        typename arg<const real>::type n,
        typename arg<const real>::type v,
        typename arg<const real>::type C,
        typename arg<const real>::type lv
)
{
	constexpr static int m = 47;
	real la = rm::log(a);
	if (a < m){
		real vshift = (v * a + m) / (a + m);
		return n*log_rising_factorial<real>(a, la, m)
		       - log_rising_factorial<real>(v*a, lv+la, m)
		       + loggamma_v_a__minus__n_loggamma_a__plus__C_a<real>(
		              a+m,
		              n,
		              vshift,
		              C * a / (a + m),
		              rm::log(vshift));
	}
	/* This function computes the term
	 *    loggamma(v*a) - n*loggamma(a)
	 * using either the lgamma function or a series expansion.
	 */
	real v2 = v*v;
	real v4 = v2*v2;
	real a2 = a*a;
	real g = a * (((n - v) * (1.0 - la) + v * lv) + C)
	         + 0.5 * ((n - 1.0) * (la - log_2_pi<real>::value) - lv)
	         + 1/(12*a) * (1/v - n
	                       + 1/(30*a2)*(n - 1/(v*v2)
	                                     + 2/(7*a2)*(1/(v*v4) - n
	                                                 + 3/(4*a2) * (n - 1/(v*v2*v4)))
	                                   )
	                      );

	/* Leading-order error estimate from the series expansion: */
	real err = rm::abs((n - 1.0/(v*v4*v4)) / (1188*a*a2*a2*a2*a2));
	if (err < std::numeric_limits<real>::epsilon() * rm::abs(g)){
		return g;
	}

	/* Otherwise compute the difference of the log-gamma functions: */
	return loggamma_v_a__minus__n_loggamma_a__plus__C_a_explicit<real>(a, n, v, C);
}

template<typename real>
real v_digamma_v_a__minus__n_digamma_a(
        typename arg<const real>::type a,
        typename arg<const real>::type n,
        typename arg<const real>::type v
)
{
	real v3 = v*v*v;
	real a2 = a*a;
	real g = (v - n) * rm::log(a) + v * rm::log(v)
	         + 1/(2*a) * (n - 1 + 1/(6*a) * (n - 1/v - (n - 1/v3)/(10*a2) ));

	/* Leading-order error estimate from the series expansion: */
	real err = rm::abs((n - 1/(v3*v*v)) / (252*a2*a2*a2));
	if (err < std::numeric_limits<real>::epsilon() * rm::abs(g)){
		return g;
	}

	/* Otherwise compute the difference of the digamma functions: */
	return v * rm::digamma(v * a) - n * rm::digamma(a);
}

template<typename real>
real v2_trigamma_v_a__minus__n_trigamma_a(
        typename arg<const real>::type a,
        typename arg<const real>::type n,
        typename arg<const real>::type v
)
{
	real g = 1/a*((v-n) + 1/(2*a)*((1-n) + 1/(3*a)*(1/v - n )));

	/* Leading-order error estimate from the series expansion: */
	real err = rm::abs((n - 1/(v*v*v)) / (30*a*a*a*a*a));
	if (err < std::numeric_limits<real>::epsilon() * rm::abs(g)){
		return g;
	}

	/* Otherwise compute the difference of the digamma functions: */
	return v * v * rm::trigamma(v * a) - n * rm::trigamma(a);
}


template<typename real>
real log_integrand_amax(typename arg<const real>::type l1pwz,
                        typename arg<const real>::type lkiz_sum,
                        const Locals<real>& L)
{
	/* Uses Newton-Raphson to compute the (approximate) maximum of the
	 * integrand (disregarding the second integral) of the normalization
	 * of the posterior.
	 */
	const real C = L.lp - L.v*(L.ls + l1pwz) + lkiz_sum;
	real a = 1.0;
	real f0, f1, da;

	/* Ensure that the limit (a -> inf) of the derivative f0 is negative:
	 * This condition looks at the term
	 *      a * (((n - v) * (1.0 - la) + v * lv) + C)
	 * of the function loggamma_v_a__minus__n_loggamma_a__plus__C_a and
	 * ensures that this converges to -infty.
	 *
	 * Note: We do not properly handle the case  C == v * lv.
	 *       This case is probably particularly rare enough not to warrant
	 *       the increased complexity of handling it. In case it occurs,
	 *       we reject it here. An improved version would then look at
	 *           0.5 * ((n - 1.0) * (la - l2pi) - lv)
	 *           + (1/v - n)/(12*a)
	 *       to decide whether the posterior is normalizeable.
	 */
	if (L.n < L.v)
		throw std::domain_error("Posterior not normalizeable because n < v, while "
			"n >= v is required."
		);
	if (L.n == L.v && (C >= 0 || -C <= L.v * rm::log(L.v))){
		throw std::domain_error("Posterior is not normalizeable. Your data set "
			"might be insufficient or lead to a degenerate case. Providing prior "
			"parameters informed by n > v may also resolve the numerical difficulty."
		);
	}

	/* Try first with Newton's method: */
	bool success = false;
	for (size_t i=0; i<30; ++i){
		// f0 = L.v * rm::digamma(L.v * a) - L.n * rm::digamma(a) + C;
		// f1 = L.v * L.v * rm::trigamma(L.v * a) - L.n * rm::trigamma(a);
		f0 = v_digamma_v_a__minus__n_digamma_a<real>(a, L.n, L.v) + C;
		f1 = v2_trigamma_v_a__minus__n_trigamma_a<real>(a, L.n, L.v);
		da = -f0 / f1;
		da = std::max<real>(std::min<real>(da, 1e3 * a), -0.999*a);
		real anext = std::max<real>(a + da, 1e-8);
		real da_real = rm::abs(a - anext);
		success = rm::abs(da_real) <= 1e-8 * a;
		a = anext;
		if (success)
			break;
	}

	if (!success){
		/* Use Brent's method as a fallback: */
		auto cost = [l1pwz, lkiz_sum, &L](typename arg<const real>::type x) -> real
		{
			real a = L.amin / x;
			if (rm::isinf(a))
				return std::numeric_limits<real>::infinity();

			real lS = loggamma_v_a__minus__n_loggamma_a__plus__C_a<real>(
						a, L.n, L.v,
						(L.lp - L.v * (L.ls + l1pwz) + lkiz_sum),
						L.lv
			) - (L.lp + lkiz_sum);

			return -lS;
		};

		std::uintmax_t max_iter = 200;
		real xmax = boost::math::tools::brent_find_minima(
						cost,
						static_cast<real>(0.0),
						static_cast<real>(1.0),
						1000000,
						max_iter).first;

		return L.amin / xmax;
	}
	return a;
}


template<typename real>
struct itgmax_t {
	real a;
	real logI;
};


template<typename real>
itgmax_t<real> log_integrand_maximum(typename arg<const real>::type l1pwz,
                                     typename arg<const real>::type lkiz_sum,
                                     const Locals<real>& L)
{
	itgmax_t<real> res;
	res.a = log_integrand_amax(l1pwz, lkiz_sum, L);

	res.logI = loggamma_v_a__minus__n_loggamma_a__plus__C_a<real>(
	                res.a, L.n, L.v,
	                (L.lp - L.v * (L.ls + l1pwz) + lkiz_sum),
	                L.lv
	           ) - (L.lp + lkiz_sum);

	//res.logI = rm::lgamma(L.v * res.a) + (res.a - 1.0) * L.lp
	//           - L.n * rm::lgamma(res.a) - L.v * res.a * (L.ls + l1pwz)
	//           + (res.a - 1) * lkiz_sum;

	return res;
}


/*
 * The innermost integrand of the double integral; the integrand in `a`.
 */
template<bool log_integrand, typename real>
real inner_integrand_template(typename arg<const real>::type a,
                              typename arg<const real>::type l1p_kiz_sum,
                              typename arg<const real>::type l1p_wz,
                              typename arg<const real>::type log_integrand_max,
                              const Locals<real>& L)
{

	/*
	 * Shortcut for small a
	 * With SymPy, we find the following limit for a -> 0:
	 *    loggamma(v*a) - v*n*loggamma(a) --> -inf * sign(n - 1)
	 * Since for finite data sets, n>1 (the number of data points
	 * will be added), we hence have lS -> -inf and thus S -> 0
	 */
	if (a == 0)
		return 0;

	real lS = loggamma_v_a__minus__n_loggamma_a__plus__C_a<real>(
	            a, L.n, L.v,
	            (L.lp - L.v * (L.ls + l1p_wz) + l1p_kiz_sum),
	            L.lv
	) - (L.lp + l1p_kiz_sum) - log_integrand_max;

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
struct outer_integrand_base_t {
	real S;
	real local_log_scale;
	real global_log_scale;

	outer_integrand_base_t(typename arg<const real>::type S,
	        typename arg<const real>::type local_log_scale,
	        typename arg<const real>::type global_log_scale
	)
	   : S(S), local_log_scale(local_log_scale),
	     global_log_scale(global_log_scale)
	{
	}
};


template<typename real>
outer_integrand_base_t<real>
outer_integrand_base(typename arg<const real>::type z, const Locals<real>& L,
                     typename arg<const real>::type log_scale,
                     typename arg<const real>::type Iref = 0)
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
	auto integrand = [&](typename arg<const real>::type a) -> real {
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
			real S0 = int0.integrate(integrand, L.amin, lImax.a,
			                         TOL_TANH_SINH, &error, &L1, &lvl0);
			real S1 = int1.integrate(integrand, lImax.a,
			                         std::numeric_limits<real>::infinity(),
			                         TOL_TANH_SINH, &error1, &L11, &lvl1);
			if (rm::isinf(S0))
				throw std::runtime_error("S0 is inf in outer_integrand.");
			if (rm::isinf(S1))
				throw std::runtime_error("S1 is inf in outer_integrand.");
			S = S0 + S1;
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

	if (S == 0.0){
		/* If S is zero, we believe that also the rescale does not change this
		 * (as the rescale should bring S to a finite value at the maximum).
		 * Return 0.
		 */
		return outer_integrand_base_t<real>(0.0, 0.0, 0.0);
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

	real res(S * rm::exp(lImax.logI - log_scale));
	if (rm::isnan(res)){
		std::string msg("Result is NaN in outer_integrand.\nS = ");
		msg += std::to_string(static_cast<long double>(S));
		msg += "\nlImax = ";
		msg += std::to_string(static_cast<long double>(lImax.logI));
		msg += "\nlog_scale = ";
		msg += std::to_string(static_cast<long double>(log_scale));
		throw std::runtime_error(msg);
	}

	return outer_integrand_base_t<real>(S, lImax.logI, log_scale);
}

template<typename real>
real outer_integrand(typename arg<const real>::type z, const Locals<real>& L,
                     typename arg<const real>::type log_scale,
                     typename arg<const real>::type Iref = 0)
{
	outer_integrand_base_t<real> base(outer_integrand_base(z, L, log_scale, Iref));

	return base.S * rm::exp(base.local_log_scale - base.global_log_scale);
}


template<typename real>
real log_outer_integrand(typename arg<const real>::type z,
                         const Locals<real>& L,
                         typename arg<const real>::type log_scale,
                         typename arg<const real>::type Iref = 0)
{
	outer_integrand_base_t<real> base(outer_integrand_base(z, L, log_scale, Iref));

	return rm::log(base.S) + base.local_log_scale - base.global_log_scale;
}


} // namespace anomaly
} // namespace posterior
} // namespace reheatfunq

#endif