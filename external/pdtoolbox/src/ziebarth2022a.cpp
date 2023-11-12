/* Code for computing the gamma conjugate posterior modified for heat
 * flow anomaly as described by Ziebarth et al. [1].
 * This code is an alternative implementation of the code found in
 * `ziebarth2021a.cpp`.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ,
 *               2022 Malte J. Ziebarth
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
 * [1] Ziebarth, Anderson, von Specht, Heidbach, Cotton (in prep.)
 */
#include <vector>
#include <map>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>
#include <limits>
#include <optional>
#include <array>
#include <numbers>
#define BOOST_ENABLE_ASSERT_HANDLER // Make sure the asserts do not abort
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>
#include <algorithm>
#include <iostream>

#include <constexpr.hpp>
#include <ziebarth2022a.hpp>
#include <quantileinverter.hpp>
#include <numerics/cdfeval.hpp>


using boost::math::digamma;
using boost::math::trigamma;
using boost::math::quadrature::exp_sinh;
using boost::math::quadrature::tanh_sinh;
using boost::math::quadrature::gauss_kronrod;
using boost::math::quadrature::gauss;
using boost::math::tools::toms748_solve;
using boost::math::tools::eps_tolerance;

using pdtoolbox::QuantileInverter;
using pdtoolbox::OR;
using pdtoolbox::AND;
using pdtoolbox::cnst_sqrt;

using reheatfunq::numerics::CDFEval;

/*
 * Use a high precision floating point format.
 */
#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
using boost::multiprecision::float128;
typedef float128 real_t;
using boost::multiprecision::cpp_dec_float_50;
using boost::multiprecision::cpp_dec_float_100;
#else
typedef long double float128;
typedef long double real_t;
using boost::multiprecision::cpp_dec_float_50;
using boost::multiprecision::cpp_dec_float_100;
#endif


#include <chrono>
#include <thread>


/* A custom exception type indicating that the integral is out of
 * scale for double precision: */
template<typename real>
class ScaleError : public std::exception
{
public:
	explicit ScaleError(const char* msg, real log_scale)
	    : _lscale(log_scale), msg(msg) {};

	virtual const char* what() const noexcept
	{
		return msg;
	}

	real log_scale() const
	{
		return _lscale;
	};

private:
	real _lscale;
	const char* msg;
};


template<typename real>
class PrecisionError : public std::exception
{
public:
	explicit PrecisionError(const char* msg, real error, real L1)
	    : _error(error), _L1(L1),
	      msg(generate_message(msg, static_cast<double>(error),
	                           static_cast<double>(L1))) {};

	virtual const char* what() const noexcept
	{
		return msg.c_str();
	}

	real error() const
	{
		return _error;
	};

	real L1() const
	{
		return _L1;
	};

	void append_message(const char* ap_msg){
		msg.append("\n");
		msg.append(ap_msg);
	}

private:
	double _error, _L1;
	std::string msg;

	static std::string generate_message(const char* msg, double error,
	                                    double L1)
	{
		std::string smsg("PrecisionError(\"");
		smsg.append(msg);
		smsg.append("\")\n  error: ");
		std::ostringstream os_err;
		os_err << std::setprecision(18);
		os_err << error;
		smsg.append(os_err.str());
		smsg.append("\n     L1: ");
		std::ostringstream os_L1;
		os_L1 << std::setprecision(18);
		os_L1 << L1;
		smsg.append(os_L1.str());
		return smsg;
	}
};


/*************************************************************
 *                                                           *
 *    1) Full quadrature of the two integrals in a and z.    *
 *                                                           *
 *************************************************************/

template<typename real>
struct locals_t {
	/* Prior parameters (potentially updated): */
	real lp;
	real ls;
	real n;
	real v;
	real amin;
	real Qmax;
	real h0;
	real h1;
	real h2;
	real h3;
	real w;
	real lh0;
	real l1p_w;
	real log_scale;
	real ztrans;
	real norm;
	real Iref;
	real full_taylor_integral;
	std::vector<real> ki;
	std::optional<CDFEval<real>> cdf_eval;
};


/*
 * Get a scale factor for the integrals by computing an approximate maximum
 * of the integrand of the alpha integral (disregarding the z-integral, i.e.
 * assuming constant z):
 */

template<typename real>
real log_integrand_amax(const real& l1pwz, const real& lkiz_sum,
                        const locals_t<real>& L)
{
	using std::abs;
	using boost::multiprecision::abs;

	/* Uses Newton-Raphson to compute the (approximate) maximum of the
	 * integrand (disregarding the second integral) of the normalization
	 * of the posterior.
	 */
	const real C = L.lp - L.v*(L.ls + l1pwz) + lkiz_sum;
	real a = 1.0;
	real f0, f1, da;

	for (size_t i=0; i<20; ++i){
		f0 = L.v * digamma(L.v * a) - L.n * digamma(a) + C;
		f1 = L.v * L.v * trigamma(L.v * a) - L.n * trigamma(a);
		da = f0 / f1;
		a -= da;
		a = std::max<real>(a, 1e-8);
		if (abs(da) <= 1e-8 * a)
			break;
	}
	return a;
}


template<typename real>
struct itgmax_t {
	real a;
	real logI;
};


template<typename real>
itgmax_t<real> log_integrand_maximum(const real& l1pwz, const real& lkiz_sum,
                                     const locals_t<real>& L)
{
	using std::lgamma;
	using boost::multiprecision::lgamma;

	itgmax_t<real> res;
	res.a = log_integrand_amax(l1pwz, lkiz_sum, L);

	res.logI = lgamma(L.v * res.a) + (res.a - 1.0) * L.lp - L.n*lgamma(res.a)
	           - L.v * res.a*(L.ls + l1pwz) + (res.a - 1) * lkiz_sum;

	return res;
}


/*
 * The innermost integrand of the double integral; the integrand in `a`.
 */
template<bool log_integrand, typename real>
real inner_integrand_template(const real a, const real& l1p_kiz_sum,
                              const real& l1p_wz, const real& log_integrand_max,
                              const locals_t<real>& L)
{
	using std::exp;
	using boost::multiprecision::exp;
	using std::lgamma;
	using boost::multiprecision::lgamma;
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::isnan;
	using boost::multiprecision::isnan;

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
	const real lga = lgamma(a);
	if (isinf(lga))
		return 0;
	const real lgva = lgamma(va);
	if (isinf(lgva))
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
	real result = exp(lS);
	if (isinf(result)){
		throw ScaleError<real>("inner_integrand", lS);
	}

	if (isnan(result)){
		std::string msg("inner_integrand: NaN result at a =");
		msg.append(std::to_string(static_cast<double>(a)));
		msg.append(". lS = ");
		msg.append(std::to_string(static_cast<double>(lS)));
		throw std::runtime_error(msg);
	}

	return result;
}


template<typename real>
real outer_integrand(const real z, const locals_t<real>& L)
{
	/* Automatically choose the right numerical functions: */
	using std::log1p;
	using boost::multiprecision::log1p;
	using std::exp;
	using boost::multiprecision::exp;
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::isnan;
	using boost::multiprecision::isnan;

	const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

	if (isnan(z))
		throw std::runtime_error("`z` is nan!");

	// Set the inner parameters:
	real l1p_kiz_sum = 0.0;
	for (const real& k : L.ki)
		l1p_kiz_sum += log1p(-k * z);
	const real l1p_wz = log1p(-L.w * z);

	/* The non-degenerate case.
	 * First compute the maximum of the a integrand at this given z:
	 */
	const itgmax_t<real> lImax = log_integrand_maximum(l1p_wz, l1p_kiz_sum, L);

	// Integrate:
	auto integrand = [&](real a) -> real {
		real result = inner_integrand_template<false>(a, l1p_kiz_sum, l1p_wz,
		                                              lImax.logI, L);
		return result;
	};

	real error, L1, S;
	try {
		/* We wrap the integration into a try-catch to be able to distinguish
		 * the source of the error if any should be thrown.
		 */
		if (lImax.a > L.amin){
			size_t lvl0, lvl1;
			real error1, L11;
			tanh_sinh<real> int0;
			exp_sinh<real> int1;
			S = int0.integrate(integrand, L.amin, lImax.a,
			                   TOL_TANH_SINH, &error, &L1, &lvl0)
			  + int1.integrate(integrand, lImax.a,
			                   std::numeric_limits<real>::infinity(),
			                   TOL_TANH_SINH, &error1, &L11, &lvl1);
			L1 += L11;
			error += error1;
		} else {
			size_t levels;
			tanh_sinh<real> integrator;
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

	if (isinf(S)){
		throw ScaleError<real>("outer_integrand", L.log_scale);
	}

	/* Error checking: */
	constexpr double ltol = std::log(1e-5);
	const real cmp_err = std::max<real>(log(L1),
	                                    log(L.Iref) + L.log_scale - lImax.logI);
	if (log(error) > ltol + cmp_err){
		/* Check if this is relevant: */
		throw PrecisionError<real>("outer_integrand", error, L1);
	}

	return S * exp(lImax.logI - L.log_scale);
}



template<typename real>
struct az_t {
	real a;
	real z;
	real log_integrand;
};

/*
 * Find the maximum of the inner integrand across all a & z
 */
template<typename real>
az_t<real> log_integrand_max(const locals_t<real>& L)
{
	using std::log1p;
	using boost::multiprecision::log1p;
	using std::lgamma;
	using boost::multiprecision::lgamma;
	using std::abs;
	using boost::multiprecision::abs;

	/* Start from the middle of the interval: */
	real z = 0.5;
	real l1p_kiz_sum = 0.0;

	for (uint_fast8_t i=0; i<200; ++i){
		/* Set the new parameters:: */
		l1p_kiz_sum = 0.0;
		real k_1mkz_sum = 0.0;
		real k2_1mkz2_sum = 0.0;
		for (const real& k : L.ki){
			l1p_kiz_sum += log1p(-k * z);
			/* First and second derivatives of the above by z: */
			real x = k / (1.0 - k*z);
			k_1mkz_sum -= x;
			k2_1mkz2_sum -= x*x;
		}

		real l1p_wz = log1p(-L.w * z);
		real w_1mwz = -L.w / (1.0 - L.w*z);
		real w2_1mwz2 = - w_1mwz * w_1mwz;


		/* New amax: */
		real amax = std::max(log_integrand_amax(l1p_wz, l1p_kiz_sum, L),
		                     L.amin);

		/* Log of integrand:
		 * f0 =   std::lgamma(v*amax) + (amax - 1.0) * lp
		 *           - n*std::lgamma(amax) - v*amax*(ls + l1p_wz)
		 *           + (amax - 1.0) * l1p_kiz_sum
		 */

		/* Derivative of the log of the integrand by z: */
		real f1 = - L.v * amax * w_1mwz + (amax - 1.0) * k_1mkz_sum;

		/* Second derivative of the log of the integrand by z: */
		real f2 = - L.v * amax * w2_1mwz2 + (amax - 1.0) * k2_1mkz2_sum;

		/* Newton step: */
		real znext = std::min<real>(std::max<real>(z - f1 / f2, 0.0),
		                            1.0 - 1e-8);
		bool exit = abs(znext - z) < 1e-8;
		z = znext;
		if (exit)
			break;
	}

	/* Determine amax for the final iteration: */
	l1p_kiz_sum = 0.0;
	for (const real& k : L.ki)
		l1p_kiz_sum += log1p(-k * z);
	real l1p_wz = log1p(-L.w * z);
	real amax = std::max(log_integrand_amax(l1p_wz, l1p_kiz_sum, L),
	                     L.amin);

	/* Log of the integrand: */
	const real f0 = lgamma(L.v * amax) + (amax - 1.0) * L.lp
	                - L.n * lgamma(amax) - L.v * amax * (L.ls + l1p_wz)
	                + (amax - 1.0) * l1p_kiz_sum;

	return {.a=amax, .z=z, .log_integrand=f0};
}





/***************************************************************************
 *                                                                         *
 * 2) The transition zone for large z, where we perform a Taylor expansion *
 *    in y=(1-z) and explicitly calculate the z integration.               *
 *    This code is structured in the following sections:                   *
 *    - compute the constants C0 - C3 occurring with the different powers  *
 *      of y.                                                              *
 *                                                                         *
 ***************************************************************************/
enum C_t {
	C0=0, C1=1, C2=2, C3=3
};

template<typename real>
struct C1_t {
	real deriv0;
	real deriv1;
	real deriv2;

	C1_t(const real a, const locals_t<real>& L)
	{
		auto h1h0 = L.h1 / L.h0;
		deriv0 = (h1h0 + L.v * L.w / (L.w - 1)) * a - h1h0;
		deriv1 = h1h0 + L.v * L.w / (L.w - 1);
		deriv2 = 0.0;
	}
};


/*
 * Constant C2:
 */

template<typename real>
struct C2_t {
	real deriv0;
	real deriv1;
	real deriv2;

	C2_t(const real a, const locals_t<real>& L)
	{
		auto h1h0 = L.h1 / L.h0;
		auto h2h0 = L.h2 / L.h0;
		auto nrm = 1.0 / (L.w * L.w - 2 * L.w + 1);
		auto D2 = (L.v * L.v * L.w * L.w
		           + h1h0 * (2 * L.v * L.w * (L.w - 1)
		                     +  h1h0 * (1 + L.w * (L.w - 2)))
		          ) * nrm;
		auto D1 = (L.v * L.w * L.w + 2 * h2h0 * (L.w * (L.w - 2) + 1)
		           - h1h0 * (2 * L.w * L.v * (L.w - 1)
		                     + 3 * h1h0 * (L.w * (L.w - 2) + 1))) * nrm;
		auto D0 = 2*(h1h0*h1h0 - h2h0);
		deriv0 = D0 + a * (D1 + a * D2);
		deriv2 = 2 * D2;
		deriv1 = deriv2 * a + D1;
	}
};



/*
 * Constant C3:
 */

template<typename real>
struct C3_t {
	real deriv0;
	real deriv1;
	real deriv2;

	C3_t(const real a, const locals_t<real>& L)
	{
		auto h1h0 = L.h1 / L.h0;
		auto h2h0 = L.h2 / L.h0;
		auto h3h0 = L.h3 / L.h0;
		real v2 = L.v * L.v;
		auto v3 = v2 * L.v;
		real w2 = L.w * L.w;
		auto w3 = w2 * L.w;
		real F = L.w * (L.w * (L.w - 3) + 3) - 1; // w3 - 3*w2 + 3*w - 1
		auto nrm = 1/F;
		auto D3 = (v3 * w3
		           + h1h0 * (3 * v2 * w2 * (L.w - 1)
		                     + h1h0 * (3 * L.v * L.w * (L.w * (L.w - 2) + 1)
		                               + h1h0 * F))
		          ) * nrm;

		auto D2 = 3 * (v2*w3 + 2 * h2h0 * L.v * L.w * (L.w - 1) * (L.w - 1)
		               + h1h0 * (L.v * w2 * (L.v - 1) * (1 - L.w)
		                         + 2 * h2h0 * F
		                         + h1h0 * (3 * L.v * L.w * (L.w * (2 - L.w) - 1)
		                                   - 2 * h1h0 * F))
		            ) * nrm;
		auto D1 = (  2 * L.v * w3  +  6 * h3h0 * F
		           + 6 * h2h0 * L.v * L.w * (L.w * (2 - L.w) - 1)
		           + h1h0 * (3 * L.v * w2 * (1 - L.w)
		                     - 18 * h2h0 * F
		                     + h1h0 * (6 * L.v * L.w * (w2 - 2 * L.w + 1)
		                               + 11 * h1h0*F))
		          ) * nrm;

		auto D0 = 6 * (h1h0 * (2 * h2h0 - h1h0*h1h0) - h3h0);

		deriv0 = D0 + a * (D1 + a * (D2 + a * D3));
		deriv1 = D1 + a * (2*D2 + a * 3 * D3);
		deriv2 = 2 * (D2 + 3 * a * D3);
	}
};


template<typename real, unsigned char order>
real large_z_amax(const real ym, const locals_t<real>& L)
{
	using std::log;
	using boost::multiprecision::log;
	using std::abs;
	using boost::multiprecision::abs;

	/* This method computes max location of the maximum of the four integrals
	 * used in the Taylor expansion of the double integral for large z. */
	constexpr double TOL = 1e-14;
	const real lh0 = log(L.h0);
	const real lym =  log(ym);
	real amax = (L.n > L.v) ? std::max<real>((L.lp - L.v * L.ls + L.v * log(L.v)
	                                          + lh0 - L.v * L.l1p_w + lym)
	                                         / (L.n - L.v),
	                                         1.0)
	                    : 1.0;

	/* Recurring terms of the first and second derivatives of the a-integrands:
	 */
	auto f0_base = [&](real a) -> real {
		return L.v * digamma(L.v * a) + L.lp - L.n * digamma(a) - L.v * L.ls
		       + lh0 - L.v * L.l1p_w;
	};
	auto f1_base = [&](real a) -> real {
		return L.v * L.v * trigamma(L.v * a) - L.n * trigamma(a);
	};

	if (order == 0){
		for (uint_fast8_t i=0; i<200; ++i){
			const real f0 = f0_base(amax) - 1/amax + lym;
			const real f1 = f1_base(amax) + 1/(amax*amax);
			const real da = std::max<real>(-f0/f1, -0.9*amax);
			amax += da;
			if (abs(da) < TOL*abs(amax))
				break;
		}
	} else if (order == 1){
		for (uint_fast8_t i=0; i<200; ++i){
			C1_t<real> C1(amax, L);
			real C1_1 = C1.deriv1 / C1.deriv0;
			const real f0 = f0_base(amax) - 1/(amax+1) + lym + C1_1;
			const real f1 = f1_base(amax) + 1/((amax+1)*(amax+1))
			                + C1.deriv2/C1.deriv0 - C1_1 * C1_1;
			const real da = std::max<real>(-f0/f1, -0.9*amax);
			amax += da;
			if (abs(da) < TOL*abs(amax))
				break;
		}
	} else if (order == 2){
		for (uint_fast8_t i=0; i<200; ++i){
			C2_t<real> C2(amax, L);
			real C2_1 = C2.deriv1/C2.deriv0;
			const real f0 = f0_base(amax) - 1/(amax+2) + lym + C2_1;
			const real f1 = f1_base(amax) + 1/((amax+2)*(amax+2))
			                + C2.deriv2/C2.deriv0 - C2_1 * C2_1;
			const real da = std::max<real>(-f0/f1, -0.9*amax);
			amax += da;
			if (abs(da) < TOL*abs(amax))
				break;
		}
	} else if (order == 3){
		for (int i=0; i<200; ++i){
			C3_t<real> C3(amax, L);
			const real C3_1 = C3.deriv1/C3.deriv0;
			auto C3_2 = C3.deriv2/C3.deriv0 - C3_1 * C3_1;
			const real f0 = f0_base(amax) - 1/(amax+3) + lym + C3_1;
			const real f1 = f1_base(amax) + 1/((amax+3)*(amax+3)) + C3_2;
			const real da = std::max<real>(-f0/f1, -0.9*amax);
			amax += da;
			if (abs(da) < TOL*abs(amax))
				break;
		}
	}
	return std::max(amax, L.amin);
}


/*
 * Integrand of the 'a'-integral given y^(a+m)
 */

template<typename real>
struct log_double_t {
	real log_abs;
	int8_t sign;
};


template<C_t C, bool y_integrated, typename real>
log_double_t<real>
a_integral_large_z_log_integrand(real a, real ly, real log_integrand_max,
                                 const locals_t<real>& L)
{
	using std::log;
	using boost::multiprecision::log;
	using std::lgamma;
	using boost::multiprecision::lgamma;
	using std::isinf;
	using boost::multiprecision::isinf;

	const real va = L.v * a;

	// Compute C:
	real lC = 0.0;
	int8_t sign = 1;
	if (C == C_t::C0){
		/* C0 = 1 */
		lC = 0.0;
	} else if (C == C_t::C1) {
		/* C1 */
		const C1_t C1(a, L);
		if (C1.deriv0 < 0){
			sign = -1;
			lC = log(-C1.deriv0);
		} else {
			lC = log(C1.deriv0);
		}
	} else if (C == C_t::C2){
		/* C2 */
		const C2_t C2(a, L);
		if (C2.deriv0 < 0){
			sign = -1;
			lC = log(-C2.deriv0);
		} else {
			lC = log(C2.deriv0);
		}
	} else if (C == C_t::C3){
		/* C3 */
		const C3_t C3(a, L);
		if (C3.deriv0 < 0){
			sign = -1;
			lC = log(-C3.deriv0);
		} else {
			lC = log(C3.deriv0);
		}
	}

	/* Check if we might want to return -inf: */
	if (a == 0)
		return {.log_abs=-std::numeric_limits<real>::infinity(),
		        .sign=sign};
	const real lgva = lgamma(va);
	const real lga = lgamma(a);
	if (isinf(lgva) || isinf(lga))
		return {.log_abs=-std::numeric_limits<real>::infinity(),
		        .sign=sign};

	// Term
	//     C * y^(a+m) / (a+m)
	// from the y power integration or
	//     C * y^(a+m-1)
	// if not integrated
	constexpr unsigned char m = C;
	if (y_integrated){
		// Integrated:
		lC += (a + m) * ly - log(a+m);
	} else {
		// Not integrated:
		lC += (a + m - 1) * ly;
	}

	// Term ( s_tilde / (1-w) ) ^ va
	lC -= va * (L.ls + L.l1p_w);

	// Remaining summands:
	lC += lgva + (a - 1.0) * (L.lp + L.lh0) - L.n * lga - log_integrand_max;

	if (isinf(lC)){
		return {.log_abs=-std::numeric_limits<double>::infinity(),
		        .sign=sign};
	}

	return {.log_abs=lC, .sign=sign};
}

template<C_t C, bool y_integrated, typename real>
real a_integral_large_z_integrand(real a, real ly, real log_integrand_max,
                                  const locals_t<real>& L)
{
	using std::exp;
	using boost::multiprecision::exp;
	using std::isinf;
	using boost::multiprecision::isinf;

	log_double_t<real> res
	    = a_integral_large_z_log_integrand<C,y_integrated, real>
	           (a, ly, log_integrand_max, L);

	// Compute the result and test for finity:
	real result = exp(res.log_abs);
	if (isinf(result)){
		if (C == C_t::C0)
			throw ScaleError<real>("a_integral_large_z_integrand_0",
			                       res.log_abs);
		else if (C == C_t::C1)
			throw ScaleError<real>("a_integral_large_z_integrand_1",
			                       res.log_abs);
		else if (C == C_t::C2)
			throw ScaleError<real>("a_integral_large_z_integrand_2",
			                       res.log_abs);
		else if (C == C_t::C3)
			throw ScaleError<real>("a_integral_large_z_integrand_3",
			                       res.log_abs);
	}

	return result;
}


template<typename real>
real y_taylor_transition_root_backend(real y, const locals_t<real>& L)
{
	using std::log;
	using boost::multiprecision::log;
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::abs;
	using boost::multiprecision::abs;

	/* Backup Gauss-Kronrod quadrature: */
	typedef gauss_kronrod<real,31> GK;

	constexpr double epsilon = 1e-14;

	const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

	/* Get the scale: */
	const real amax = large_z_amax<real,0>(y, L);
	const real ly = log(y);
	const real log_scale = L.log_scale
	    + a_integral_large_z_log_integrand<C_t::C0,true>(amax, ly, L.log_scale,
	                                                     L).log_abs;

	/* Compute the 'a' integrals for the constant and the cubic term: */
	real error, L1;
	size_t levels;
	exp_sinh<real> es_integrator;
	tanh_sinh<real> ts_integrator;

	auto integrand00 = [&](real a, real distance_to_border) -> real
	{
		return a_integral_large_z_integrand<C_t::C0,true,real>(
		                a, ly, log_scale, L
		);
	};

	auto integrand0 = [&](real a) -> real
	{
		return a_integral_large_z_integrand<C_t::C0,true,real>(
		                a, ly, log_scale, L
		);
	};

	real S0;
	if (amax > L.amin){
		real error1, L11;
		/* Try tanh_sinh for the integral [amin, amax] first, but fall back
		 * to Gauss-Kronrod if an exception is thrown:
		 */
		try {
			S0 = ts_integrator.integrate(integrand00, L.amin, amax,
			                             TOL_TANH_SINH, &error, &L1, &levels);
		} catch (...) {
			S0 = GK::integrate(integrand0, L.amin, amax, 9, TOL_TANH_SINH,
			                   &error, &L1);
		}
		S0 += es_integrator.integrate(integrand0, amax,
		                              std::numeric_limits<real>::infinity(),
		                              TOL_TANH_SINH, &error1, &L11, &levels);
		error += error1;
		L1 += L11;
	} else {
		S0 = es_integrator.integrate(integrand0, L.amin,
		                             std::numeric_limits<real>::infinity(),
		                             TOL_TANH_SINH, &error, &L1, &levels);
	}

	if (isinf(S0)){
		throw ScaleError<real>("y_taylor_transition_root_backend_S0",
		                       log_scale);
	}
	/* Error checking: */
	if (error > 1e-2 * L1){
		throw PrecisionError<real>("y_taylor_transition_root_backend_S0",
		                           error, L1);
	}

	auto integrand10 = [&](real a, real distance_to_border) -> real
	{
		return a_integral_large_z_integrand<C_t::C3,true,real>(
		                a, ly, log_scale, L
		);
	};

	auto integrand1 = [&](real a) -> real
	{
		return a_integral_large_z_integrand<C_t::C3,true,real>(
		                a, ly, log_scale, L);
	};


	real S1;
	if (amax > L.amin){
		real error1, L11;
		/* Try tanh_sinh for the integral [amin, amax] first, but fall back
		 * to Gauss-Kronrod if an exception is thrown:
		 */
		try {
			S1 = ts_integrator.integrate(integrand10, L.amin, amax,
			                             TOL_TANH_SINH, &error, &L1, &levels);
		} catch (...) {
			S1 = GK::integrate(integrand1, L.amin, amax, 9, TOL_TANH_SINH,
			                   &error, &L1);
		}
		S1 += es_integrator.integrate(integrand1, amax,
		                              std::numeric_limits<real>::infinity(),
		                              TOL_TANH_SINH, &error1, &L11, &levels);
		error += error1;
		L1 += L11;
	} else {
		S1 = es_integrator.integrate(integrand1, L.amin,
		                             std::numeric_limits<real>::infinity(),
		                             TOL_TANH_SINH, &error, &L1, &levels);
	}

	if (isinf(S1)){
		throw ScaleError<real>("y_taylor_transition_root_backend_S1",
		                       log_scale);
	}
	/* Error checking: */
	if (error > 1e-2 * std::max<real>(L1,abs(S0))){
		throw PrecisionError<real>("y_taylor_transition_root_backend_S1",
		                           error, L1);
	}

	/* Extract the result: */
	const real result = log(abs(S1)) - log(epsilon)
	                    - log(abs(S0));

	/* Make sure that result is finite: */
	if (isinf(result)){
		throw ScaleError<real>("y_taylor_transition_root_backend", 300.);
	}

	return result;
}




template<typename real>
real y_taylor_transition(const locals_t<real>& L, const real ymin = 1e-32)
{
	using std::log;
	using boost::multiprecision::log;
	using std::isnan;
	using boost::multiprecision::isnan;

	/* Find a value above the threshold: */
	real yr = 1e-9;
	real val = y_taylor_transition_root_backend<real>(yr, L);
	while (val < 0 || isnan(val)){
		yr = std::min<real>(2*yr, 1.0);
		if (yr == 1.0)
			break;
		val = y_taylor_transition_root_backend<real>(yr, L);
	}

	/* Root finding: */
	auto rootfun = [&](real y) -> real {
		return y_taylor_transition_root_backend<real>(y, L);
	};
	constexpr std::uintmax_t MAX_ITER = 100;
	std::uintmax_t max_iter = MAX_ITER;
	eps_tolerance<real> tol(2);

	std::pair<real,real> bracket
	   = toms748_solve(rootfun, (real)ymin, (real)yr, tol, max_iter);

	if (max_iter >= MAX_ITER)
		throw std::runtime_error("Could not determine root.");

	return 0.5 * (bracket.first + bracket.second);
}



/*
 * Compute the actual integral:
 */
template<bool y_integrated, typename real>
real a_integral_large_z(const real ym, const real S_cmp,
                        const locals_t<real>& L)
{
	using std::log;
	using boost::multiprecision::log;
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::abs;
	using boost::multiprecision::abs;
	using std::isnan;
	using boost::multiprecision::isnan;

	const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

	/* Set the integrand's non-varying parameters: */
	const real ly = log(ym);
	const real lScmp = log(S_cmp);

	/* Integration setup for 'a' integrals:: */
	real error, L1, S;
	size_t levels;
	exp_sinh<real> integrator;

	auto integrand = [&](real a) -> real
	{
		real S0
		   = a_integral_large_z_integrand<C_t::C0,y_integrated,real>
		         (a, ly, lScmp, L);
		real S1
		   = a_integral_large_z_integrand<C_t::C1,y_integrated,real>
		         (a, ly, lScmp, L);
		real S2
		   = a_integral_large_z_integrand<C_t::C2,y_integrated,real>
		         (a, ly, lScmp, L);
		real S3
		   = a_integral_large_z_integrand<C_t::C3,y_integrated,real>
		         (a, ly, lScmp, L);
		return S0 + S1 + S2 + S3;
	};

	try {
		S = integrator.integrate(integrand, L.amin,
		                         std::numeric_limits<real>::infinity(),
		                         TOL_TANH_SINH, &error, &L1, &levels);
	} catch (std::runtime_error& e) {
		std::string msg("Error in a_integral_large_z exp_sinh routine: '");
		msg += e.what();
		msg += "'.";
		throw std::runtime_error(msg);
	}
	if (isinf(S) || isnan(S))
		throw ScaleError<real>("a_integral_large_z", L.log_scale);
	if (error > 1e-14 * std::max(L1, S_cmp))
		throw PrecisionError<real>("a_integral_large_z_S3", error, L1);

	return S;
}




/*
 * Compute the posterior:
 */

using pdtoolbox::heatflow::posterior_t;

#pragma GCC push_options
#pragma GCC optimize("-fno-associative-math")
template<typename real>
real kahan_sum(const double* x, size_t N)
{
	real S = 0.0;
	real corr = 0.0;
	for (size_t i=0; i<N; ++i){
		real dS = static_cast<real>(x[i]) - corr;
		real next = S + dS;
		corr = (next - S) - dS;
		S = next;
	}
	return S;
}

template<typename real>
real kahan_sum(const std::vector<real>& x)
{
	real S = 0.0;
	real corr = 0.0;
	for (const real& xi : x){
		real dS = xi - corr;
		real next = S + dS;
		corr = (next - S) - dS;
		S = next;
	}
	return S;
}
#pragma GCC pop_options


template<typename real, bool need_norm=true, bool use_cdf_eval=false>
void init_locals(const double* qi, const double* ci, size_t N,
                 const real v, const real n, const real s, const real p,
                 const real amin, const double* x, const size_t Nx,
                 // Destination parameters:
                 locals_t<real>& L, size_t& imax, double dest_tol)
{
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::log;
	using boost::multiprecision::log;
	using std::log1p;
	using boost::multiprecision::log1p;

	const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

	// Initialize parameters:
	L.v = v + N;
	L.n = n + N;
	L.amin = amin;

	// Step 0: Determine Qmax, A, B, and ki:
	imax = 0;
	L.Qmax = std::numeric_limits<real>::infinity();
	for (size_t i=0; i<N; ++i){
		if (qi[i] <= 0)
			throw std::runtime_error("At least one qi is zero or negative and "
			                         "has hence left the model definition "
			                         "space.");
		real Q = qi[i] / ci[i];
		if (Q > 0 && Q < L.Qmax){
			L.Qmax = Q;
			imax = i;
		}
	}
	if (isinf(L.Qmax))
		throw std::runtime_error("Found Qmax == inf. The model is not "
		                         "well-defined. This might happen if all "
		                         "ci <= 0, i.e. heat flow anomaly has a "
		                         "negative or no impact on all data points.");


	real A = s + kahan_sum<real>(qi, N);
	L.ls = log(A);
	real B = kahan_sum<real>(ci, N);
	L.ki.resize(N);
	for (size_t i=0; i<N; ++i)
		L.ki[i] = (ci[i] * L.Qmax) / qi[i];
	real lq_sum = 0.0;
	std::vector<real> lqi(N);
	for (size_t i=0; i<N; ++i)
		lqi[i] = log(qi[i]);
	lq_sum = kahan_sum(lqi);
	lqi.clear();

	L.lp = log(p) + lq_sum;
	L.w = B * L.Qmax / A;

	// Integration config:

	/* Get an estimate of the scaling: */

	/* The new version: */
	az_t azmax = log_integrand_max(L);
	L.log_scale = azmax.log_integrand;

	/* Compute the coefficients for the large-z (small-y) Taylor expansion: */
	L.h0=1.0;
	L.h1=0;
	L.h2=0;
	L.h3=0;
	for (size_t i=0; i<N; ++i){
		if (i == imax)
			continue;
		const real d0 = 1.0 - L.ki[i];
		L.h3 = d0 * L.h3 + L.ki[i] * L.h2;
		L.h2 = d0 * L.h2 + L.ki[i] * L.h1;
		L.h1 = d0 * L.h1 + L.ki[i] * L.h0;
		L.h0 *= d0;
	}
	L.lh0 = log(L.h0);

	/* Step 1: Compute the normalization constant.
	 *         This requires the full integral and might require
	 *         a readjustment of the log_scale parameter. */
	L.l1p_w = log1p(-L.w);
	bool norm_success = false;
	L.full_taylor_integral = std::nan("");
	L.norm = std::nan("");
	L.ztrans = std::nan("");
	real ymax, S;

	while (!norm_success && !isinf(L.log_scale)){
		// Compute the transition z where we switch to an analytic integral of
		// the Taylor expansion:
		try {
			ymax = y_taylor_transition(L, static_cast<real>(1e-32));
		} catch (ScaleError<real>& s) {
			if (s.log_scale() < 0){
				std::string msg("Failed to determine the normalization "
				                "constant: Encountered negative scale error "
				                "in ");
				msg.append(s.what());
				msg.append(": ");
				msg.append(std::to_string(static_cast<double>(s.log_scale())));
				throw std::runtime_error(msg);
			}
			L.log_scale += s.log_scale();
			continue;
		} catch (PrecisionError<real>& s){
			std::string msg("Failed to determine the normalization "
			                "constant: Encountered precision error "
			                "in ");
			msg.append(s.what());
			msg.append(": error=");
			msg.append(std::to_string(static_cast<double>(s.error())));
			msg.append(", L1=");
			msg.append(std::to_string(static_cast<double>(s.L1())));
			throw std::runtime_error(msg);
		}
		L.ztrans = 1.0 - ymax;

		/*
		 * For the unnormed posterior, we do not need the normalization
		 * constant.
		 */
		if (!need_norm)
			return;

		try {
			/* Compute a reference integral along the a axis at z that
			 * corresponds to the maximum (a,z): */
			L.Iref = 0;
			L.Iref = outer_integrand(azmax.z, L);


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
			if (use_cdf_eval){
				/* This is called for the CDF: */
				std::vector<real> z;
				z.reserve(Nx);
				for (size_t i=0; i<Nx; ++i){
					real zi = static_cast<real>(x[i]) / L.Qmax;
					if (zi <= L.ztrans)
						z.push_back(zi);
				}
				L.cdf_eval = CDFEval<real>(z, integrand, 0.0, L.ztrans,
				                           0.0, dest_tol);
				S = L.cdf_eval->norm();

			} else {
				real error;
				tanh_sinh<real> integrator;
				try {
					S = integrator.integrate(integrand, static_cast<real>(0),
					                         L.ztrans, TOL_TANH_SINH, &error);
				} catch (...) {
					/* Backup Gauss-Kronrod: */
					typedef gauss_kronrod<real,31> GK;
					real L1;
					S =  GK::integrate(integrand, static_cast<real>(0),
					                   L.ztrans, 9, TOL_TANH_SINH,
					                   &error, &L1);
				}
			}
		} catch (ScaleError<real>& s) {
			if (s.log_scale() < 0)
				throw std::runtime_error("Failed to determine the "
				                         "normalization constant: Encountered "
				                         "negative scale error.");
			L.log_scale += s.log_scale();
			continue;
		} catch (PrecisionError<real>& s) {
			s.append_message("Failed to determine the normalization "
			                 "constant (moderate z).");
			throw s;
		} catch (std::runtime_error& err){
			std::string msg("Failed to determine the normalization constant "
			                "for the integrand: '");
			msg.append(err.what());
			msg.append("'.\nMore information: ztrans=");
			msg.append(std::to_string(static_cast<double>(L.ztrans)));
			msg.append(", ymax=");
			std::ostringstream os_ymax;
			os_ymax << ymax;
			msg.append(os_ymax.str());
			throw std::runtime_error(msg);
		}

		// Integrate the normalization constant:
		try {
			// 1.2: Combined analytical and numerical integration for
			//      z in range [1-ymax, 1]:
			L.full_taylor_integral =
			a_integral_large_z<true>(ymax, S, L);
			S += L.full_taylor_integral;

			if (use_cdf_eval)
				L.cdf_eval->add_outside_mass(L.full_taylor_integral / S);

		} catch (ScaleError<real>& s) {
			if (s.log_scale() < 0)
				throw std::runtime_error("Failed to determine the "
				                         "normalization constant: Encountered "
				                         "negative scale error.");
			L.log_scale += s.log_scale();
			continue;
		} catch (PrecisionError<real>& s) {
			s.append_message("Failed to determine the normalization constant "
			                 "(large z).");
			throw s;
		} catch (std::runtime_error& err){
			std::string msg("Failed to determine the normalization constant "
			                "for the integrand: '");
			msg.append(err.what());
			msg.append("'.\nMore information: ztrans=");
			msg.append(std::to_string(static_cast<double>(L.ztrans)));
			msg.append(", ymax=");
			std::ostringstream os_ymax;
			os_ymax << ymax;
			msg.append(os_ymax.str());
			throw std::runtime_error(msg);
		}
		L.norm = S;
		norm_success = true;
	}
}



/*
 *
 *   Computing the posterior.
 *
 */

template<posterior_t type, typename real>
void posterior(const double* x, long double* res, size_t Nx, const double* qi,
               const double* ci, size_t N, double p, double s, double n,
               double nu, double amin, double dest_tol)
{
	using std::log;
	using boost::multiprecision::log;
	using std::abs;
	using boost::multiprecision::abs;
	using std::isnan;
	using boost::multiprecision::isnan;

	/* Computes the posterior for a given parameter combination using
	 * two-dimensional adaptive quadrature.
	 */

	constexpr bool is_cumulative
	   = type & (posterior_t::CUMULATIVE | posterior_t::TAIL);

	/* Step 1: Use the common parameter initialization routine which
	 *         mainly determines the normalization constant: */
	locals_t<real> L;
	size_t imax;
	if (type == posterior_t::UNNORMED_LOG){
		init_locals<real,false>(qi, ci, N, (real)nu, (real)n, (real)s, (real)p,
		                        (real)amin, x, Nx, L, imax, dest_tol);
	} else if (is_cumulative){
		init_locals<real,true,true>(qi, ci, N, (real)nu, (real)n, (real)s,
		                             (real)p, (real)amin, x, Nx, L, imax,
		                             dest_tol);
	} else {
		init_locals(qi, ci, N, (real)nu, (real)n, (real)s, (real)p, (real)amin,
		            x, Nx, L, imax, dest_tol);
	}

	std::vector<size_t> order(0);
	if (is_cumulative){
		order.resize(Nx);
		for (size_t i=0; i<Nx; ++i){
			order[i] = i;
		}
		if (type == posterior_t::CUMULATIVE){
			std::sort(order.begin(), order.end(),
			          [&](size_t i, size_t j) -> bool
			          {
			              return x[i] < x[j];
			          }
			);
		} else {
			std::sort(order.begin(), order.end(),
			          [&](size_t i, size_t j) -> bool
			          {
			              return x[i] > x[j];
			          }
			);
		}
	}

	auto integrand = [&](real z) -> real {
		return outer_integrand<real>(z, L);
	};

	/* Step 2: Compute the values: */
	for (size_t i=0; i<Nx; ++i){
		const size_t j = (is_cumulative) ? order[i] : i;
		// Short cut: x out of bounds:
		if (x[j] < 0){
			if (type == posterior_t::UNNORMED_LOG)
				res[j] = -std::numeric_limits<long double>::infinity();
			else if (type == posterior_t::TAIL)
				res[j] = 1.0;
			else
				res[j] = 0.0;
			continue;
		} else if (x[j] >= L.Qmax){
			if (type == posterior_t::DENSITY || type == posterior_t::TAIL)
				res[j] = 0.0;
			else if (type == posterior_t::UNNORMED_LOG)
				res[j] = -std::numeric_limits<long double>::infinity();
			else
				res[j] = 1.0;
			continue;
		}

		/* Continue depending on chosen representation of the posterior: */
		const real zi = static_cast<real>(x[j]) / L.Qmax;
		if (isnan(zi)){
			res[j] = std::nan("");
			continue;
		}

		if (type == posterior_t::DENSITY){
			/* Return the density in P_H (aka \dot{Q}) of the posterior.
			 * To do so, return the outer integrand at z corresponding to the
			 * requested x[i].
			 * Then norm the outer_integrand to a PDF in z. Revert the change of
			 * variables x->z by dividing the PDF by Qmax.
			 */
			if (zi <= L.ztrans){
				real resj = integrand(zi) / (L.Qmax * L.norm);
				res[j] = static_cast<long double>(resj);
			} else {
				// Use the Taylor expansion for the limit z->1, y=1-z -> 0:
				real resj = a_integral_large_z<false,real>(1.0 - zi, L.norm, L)
				            / (L.Qmax * L.norm);
				res[j] = static_cast<long double>(resj);
			}
		} else if (type == posterior_t::CUMULATIVE){
			real resj;
			if (zi <= L.ztrans){
				resj = L.cdf_eval->cdf(zi);
			} else {
				/* Now integrate from the back: */
				real back = a_integral_large_z<true, real>(1.0-zi, L.norm, L);
				resj = (L.norm - back) / L.norm;
			}
			resj = std::max<real>(std::min<real>(resj, 1.0), 0.0);
			res[j] = static_cast<long double>(resj);
		} else if (type == posterior_t::TAIL){
			real resj;
			if (zi <= L.ztrans){
				// First the part from zi to ztrans:
				resj = L.cdf_eval->tail(zi);
			} else {
				// The part from zi to 1:
				resj = a_integral_large_z<true, real>(1.0-zi, L.norm,
				                                      L) / L.norm;
			}
			resj = std::max<real>(std::min<real>(resj, 1.0), 0.0);
			res[j] = static_cast<long double>(resj);

		} else if (type == posterior_t::UNNORMED_LOG){
			/* Return the logarithm of the unnormed density.
			 * This mode is useful if external code is used to superpose
			 * additional dimensions that need to be taken into account
			 * when normalizing.
			 */
			real resj;
			if (zi <= L.ztrans){
				resj = log(integrand(zi)) + L.log_scale;
			} else {
				resj = log(a_integral_large_z<false,real>(1.0-zi, 0.0, L))
				         + L.log_scale;
			}
			res[j] = static_cast<long double>(resj);
		}
	}
}


/*
 * Template instantiations:
 */
namespace pdtoolbox {
namespace heatflow {

void posterior_pdf(const double* x, long double* res, size_t Nx,
                   const double* qi, const double* ci, size_t N, double p,
                   double s, double n, double nu, double amin, double dest_tol,
                   precision_t working_precision)
{
	if (working_precision == WP_DOUBLE)
		posterior<DENSITY,double>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                          dest_tol);
	else if (working_precision == WP_LONG_DOUBLE)
		posterior<DENSITY,long double>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                               dest_tol);
	else if (working_precision == WP_FLOAT_128){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
		posterior<DENSITY,float128>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                            dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_QUAD'.");
		#endif
	} else if (working_precision == WP_BOOST_DEC_50)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
		posterior<DENSITY,cpp_dec_float_50>(x, res, Nx, qi, ci, N, p, s, n, nu,
		                                    amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50'.");
		#endif
	else if (working_precision == WP_BOOST_DEC_100)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
		posterior<DENSITY,cpp_dec_float_100>(x, res, Nx, qi, ci, N, p, s, n, nu,
		                                     amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100'.");
		#endif
	else
		throw std::runtime_error("Unknown working_precision parameter.");
}


void posterior_pdf_batch(const double* x, size_t Nx, long double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double amin, double dest_tol,
                         precision_t working_precision)
{
	/* Sanity: */
	if (qi.size() != ci.size())
		throw std::runtime_error("Sizes of `qi` and `ci` do not match.");
	if (qi.size() != N.size())
		throw std::runtime_error("Sizes of `qi` and `N` do not match.");

	bool error_flag = false;
	std::string err_msg;

	#pragma omp parallel for
	for (size_t i=0; i<qi.size(); ++i){
		if (!error_flag){
			try {
				posterior_pdf(x, res + Nx*i, Nx, qi[i], ci[i], N[i], p, s, n,
				              nu, amin, dest_tol, working_precision);
			} catch (const std::exception& e){
				error_flag = true;
				err_msg = std::string(e.what());
			}
		}
	}

	if (error_flag){
		err_msg = std::string("Error in posterior_pdf_batch: '")
		          + err_msg + std::string("'.");
		throw std::runtime_error(err_msg);
	}
}


void posterior_cdf(const double* x, long double* res, size_t Nx,
                   const double* qi, const double* ci, size_t N, double p,
                   double s, double n, double nu, double amin, double dest_tol,
                   precision_t working_precision)
{
	if (working_precision == WP_DOUBLE)
		posterior<CUMULATIVE,double>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                          dest_tol);
	else if (working_precision == WP_LONG_DOUBLE)
		posterior<CUMULATIVE,long double>(x, res, Nx, qi, ci, N, p, s, n, nu,
		                                  amin, dest_tol);
	else if (working_precision == WP_FLOAT_128){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
		posterior<CUMULATIVE,float128>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                               dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_QUAD'.");
		#endif
	} else if (working_precision == WP_BOOST_DEC_50)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
		posterior<CUMULATIVE,cpp_dec_float_50>(x, res, Nx, qi, ci, N, p, s, n,
		                                       nu, amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50'.");
		#endif
	else if (working_precision == WP_BOOST_DEC_100)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
		posterior<CUMULATIVE,cpp_dec_float_100>(x, res, Nx, qi, ci, N, p, s, n,
		                                        nu, amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100'.");
		#endif
	else
		throw std::runtime_error("Unknown working_precision parameter.");
}


void posterior_cdf_batch(const double* x, size_t Nx, long double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double amin, double dest_tol,
                         precision_t working_precision)
{
	/* Sanity: */
	if (qi.size() != ci.size())
		throw std::runtime_error("Sizes of `qi` and `ci` do not match.");
	if (qi.size() != N.size())
		throw std::runtime_error("Sizes of `qi` and `N` do not match.");

	bool error_flag = false;
	std::string err_msg;

	#pragma omp parallel for
	for (size_t i=0; i<qi.size(); ++i){
		if (!error_flag){
			try {
				posterior_cdf(x, res + Nx*i, Nx, qi[i], ci[i], N[i], p, s, n,
				              nu, amin, dest_tol, working_precision);
			} catch (const std::exception& e){
				error_flag = true;
				err_msg = std::string(e.what());
			}
		}
	}

	if (error_flag){
		err_msg = std::string("Error in posterior_cdf_batch: '")
		          + err_msg + std::string("'.");
		throw std::runtime_error(err_msg);
	}
}


void posterior_tail(const double* x, long double* res, size_t Nx,
                    const double* qi, const double* ci, size_t N, double p,
                    double s, double n, double nu, double amin, double dest_tol,
                    precision_t working_precision)
{
	if (working_precision == WP_DOUBLE)
		posterior<TAIL,double>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                          dest_tol);
	else if (working_precision == WP_LONG_DOUBLE)
		posterior<TAIL,long double>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                            dest_tol);
	else if (working_precision == WP_FLOAT_128){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
		posterior<TAIL,float128>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                         dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_QUAD'.");
		#endif
	} else if (working_precision == WP_BOOST_DEC_50)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
		posterior<TAIL,cpp_dec_float_50>(x, res, Nx, qi, ci, N, p, s, n, nu,
		                                 amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50'.");
		#endif
	else if (working_precision == WP_BOOST_DEC_100)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
		posterior<TAIL,cpp_dec_float_100>(x, res, Nx, qi, ci, N, p, s, n, nu,
		                                  amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100'.");
		#endif
	else
		throw std::runtime_error("Unknown working_precision parameter.");
}


void posterior_tail_batch(const double* x, size_t Nx, long double* res,
                          const std::vector<const double*>& qi,
                          const std::vector<const double*>& ci,
                          const std::vector<size_t>& N,
                          double p, double s, double n, double nu,
                          double amin, double dest_tol,
                          precision_t working_precision)
{
	/* Sanity: */
	if (qi.size() != ci.size())
		throw std::runtime_error("Sizes of `qi` and `ci` do not match.");
	if (qi.size() != N.size())
		throw std::runtime_error("Sizes of `qi` and `N` do not match.");

	bool error_flag = false;
	std::string err_msg;

	#pragma omp parallel for
	for (size_t i=0; i<qi.size(); ++i){
		if (!error_flag){
			try {
				posterior_tail(x, res + Nx*i, Nx, qi[i], ci[i], N[i], p, s, n,
				               nu, amin, dest_tol, working_precision);
			} catch (const std::exception& e){
				error_flag = true;
				err_msg = std::string(e.what());
			}
		}
	}

	if (error_flag){
		err_msg = std::string("Error in posterior_tail_batch: '")
		          + err_msg + std::string("'.");
		throw std::runtime_error(err_msg);
	}
}


void posterior_log_unnormed(const double* x, long double* res, size_t Nx,
                            const double* qi, const double* ci, size_t N,
                            double p, double s, double n, double nu,
                            double amin, double dest_tol,
                            precision_t working_precision)
{
	if (working_precision == WP_DOUBLE)
		posterior<UNNORMED_LOG,double>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
		                               dest_tol);
	else if (working_precision == WP_LONG_DOUBLE)
		posterior<UNNORMED_LOG,long double>(x, res, Nx, qi, ci, N, p, s, n, nu,
		                                    amin, dest_tol);
	else if (working_precision == WP_FLOAT_128){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
		posterior<UNNORMED_LOG,float128>(x, res, Nx, qi, ci, N, p, s, n, nu,
		                                 amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_QUAD'.");
		#endif
	} else if (working_precision == WP_BOOST_DEC_50)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
		posterior<UNNORMED_LOG,cpp_dec_float_50>(x, res, Nx, qi, ci, N, p, s, n,
		                                         nu, amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50'.");
		#endif
	else if (working_precision == WP_BOOST_DEC_100)
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
		posterior<UNNORMED_LOG,cpp_dec_float_100>(x, res, Nx, qi, ci, N, p, s,
		                                          n, nu, amin, dest_tol);
		#else
		throw std::runtime_error("Need to recompile with define 'REHEATFUNQ_"
		                         "ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100'.");
		#endif
	else
		throw std::runtime_error("Unknown working_precision parameter.");
}


/*
 * A version of the above catching errors and returning NaNs:
 */
void posterior_silent(const double* x, long double* res, size_t Nx,
                      const double* qi, const double* ci, size_t N,
                      double p, double s, double n, double nu, double amin,
                      double dest_tol, posterior_t type,
                      precision_t working_precision)
{
	try {
		if (type == DENSITY)
			posterior_pdf(x, res, Nx, qi, ci, N, p, s, n, nu, amin, dest_tol,
			              working_precision);
		else if (type == CUMULATIVE)
			posterior_cdf(x, res, Nx, qi, ci, N, p, s, n, nu, amin, dest_tol,
			              working_precision);
		else if (type == TAIL)
			posterior_tail(x, res, Nx, qi, ci, N, p, s, n, nu, amin, dest_tol,
			               working_precision);
		else if (type == UNNORMED_LOG)
			posterior_log_unnormed(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
			                       dest_tol, working_precision);
	} catch (...) {
		for (size_t i=0; i<Nx; ++i){
			res[i] = std::numeric_limits<double>::quiet_NaN();
		}
	}
}

/*
 *
 *  Tail quantiles.
 *
 */
void tail_quantiles(const double* quantiles, double* res, const size_t Nquant,
                    const double* qi, const double* ci, const size_t N,
                    const double p, const double s, const double n,
                    const double nu, const double amin, const double dest_tol)
{
	/* Estimates the tail quantiles using the QuantileInverter class. */
	typedef long double real;

	/* Step 1: Use the common parameter initialization routine which
	 *         mainly determines the normalization constant: */
	locals_t<real> L;
	size_t imax;
	init_locals<real>(qi, ci, N, nu, n, s, p, amin, nullptr, 0, L, imax,
	                  dest_tol);


	auto integrand = [&](real z) -> real {
		real result = 0.0;
		if (z <= L.ztrans)
			result = outer_integrand<real>(z, L) / L.norm;
		else if (z < 1)
			result = a_integral_large_z<false,real>(1.0-z, L.norm, L) / L.norm;

		return result;
	};

	size_t i=0;
	try {
		/* Initialize the quantile inverter: */
		QuantileInverter<real> qinv(integrand, 0.0, 1.0, dest_tol, dest_tol,
		                            OR);

		/* Now invert: */
		for (; i<Nquant; ++i){
			real qi = static_cast<real>(1.0) - quantiles[i];
			res[i] = static_cast<double>(L.Qmax * qinv.invert(qi));
		}
	} catch (...) {
		tanh_sinh<double> integrator;
		for (; i<Nquant; ++i){
			const double zi = 1.0 - quantiles[i];
			auto rootfun = [&](double z){
				return integrator.integrate(integrand, z, 1.0) - zi;
			};
			auto tolerance = [&](double a, double b) -> bool {
				return std::fabs(a-b) <= dest_tol;
			};
			std::uintmax_t maxiter = 50;
			std::pair<double,double> bracket
			   = toms748_solve(rootfun, 0.0, 1.0, 1.0-zi, -zi, tolerance,
			                   maxiter);
			res[i] = L.Qmax * 0.5 * (bracket.first + bracket.second);
		}
	}
}


void posterior_tail_quantiles_batch(
                     const double* quantiles, double* res, const size_t Nquant,
                     const std::vector<const double*>& qi,
                     const std::vector<const double*>& ci,
                     const std::vector<size_t>& N,
                     double p, double s, double n, double nu,
                     double amin, double dest_tol)
{
	using std::abs;
	using boost::multiprecision::abs;

	typedef long double real;

	/* Sanity: */
	if (qi.size() != ci.size())
		throw std::runtime_error("Sizes of `qi` and `ci` do not match.");
	if (qi.size() != N.size())
		throw std::runtime_error("Sizes of `qi` and `N` do not match.");

	/* Capturing exceptions in OMP sections: */
	bool error_flag = false;
	std::string err_msg;

	/* Step 1: Use the common parameter initialization routine which
	 *         mainly determines the normalization constants.
	 *         Do so in parallel since we need one set of parameters per
	 *         (qi,ci) sample.
	 */
	std::vector<locals_t<real>> locals(qi.size());
	#pragma omp parallel for
	for (size_t i=0; i<qi.size(); ++i){
		if (!error_flag){
			size_t imax;
			try {
				init_locals<real>(qi[i], ci[i], N[i], nu, n, s, p, amin,
				                  nullptr, 0, locals[i], imax, dest_tol);
			} catch (const std::exception& e){
				error_flag = true;
				err_msg = std::string(e.what());
			}
		}
	}

	if (error_flag){
		err_msg = std::string("Error in posterior_tail_batch: '")
		          + err_msg + std::string("'.");
		throw std::runtime_error(err_msg);
	}

	/* The overall maximum frictional power: */
	real Qmax = 0.0;
	for (size_t i=0; i<qi.size(); ++i){
		Qmax = std::max(Qmax, locals[i].Qmax);

		/* The 'norm' parameter is calculated for z in the range [0,1].
		 * To be able to integrate over the full frictional power space,
		 * need to renorm:
		 */
		locals[i].norm *= locals[i].Qmax;
	}

	auto integrand = [&](real P_H) -> real {
		real result = 0.0;
		bool fail = false;
		#pragma omp parallel for
		for (size_t i=0; i<qi.size(); ++i){
			if (fail || P_H >= locals[i].Qmax)
				continue;
			const real z = P_H / locals[i].Qmax;
			try {
				if (z <= locals[i].ztrans)
					result += outer_integrand(z, locals[i])
					          / locals[i].norm;
				else if (z < 1)
					result += a_integral_large_z<false>(1.0 - z, locals[i].norm,
					                                    locals[i])
					          / locals[i].norm;
			} catch (...) {
				fail = true;
			}
		}

		return result / qi.size();
	};

	size_t i=0;
	try {
		/* Initialize the quantile inverter: */
		QuantileInverter<real> qinv(integrand, 0.0, Qmax, dest_tol, dest_tol,
		                            OR);

		/* Now invert: */
		for (; i<Nquant; ++i){
			res[i] = static_cast<double>(qinv.invert(1.0 - quantiles[i]));
		}
	} catch (...) {
		tanh_sinh<double> integrator;
		for (; i<Nquant; ++i){
			auto rootfun = [&](double P_H){
				return integrator.integrate(integrand, P_H, Qmax)
				       - quantiles[i];
			};
			auto tolerance = [&](double a, double b) -> bool {
				return abs(a-b) <= dest_tol;
			};
			std::uintmax_t maxiter = 50;
			std::pair<double, double> bracket
			   = toms748_solve(rootfun, 0.0, static_cast<double>(Qmax),
			                   1.0-quantiles[i], -quantiles[i], tolerance,
			                   maxiter);
			res[i] = 0.5 * (bracket.first + bracket.second);
		}
	}
}

/*
 * Same purpose as the above function but different algorithm: Evaluate the
 * tail distribution explicitly at Chebyshev points and perform a barycentric
 * Lagrange interpolation to
 */
void posterior_tail_quantiles_batch_barycentric_lagrange(
                     const double* quantiles, double* res, const size_t Nquant,
                     const std::vector<const double*>& qi,
                     const std::vector<const double*>& ci,
                     const std::vector<size_t>& N,
                     double p, double s, double n, double nu,
                     double amin, double dest_tol, precision_t precision,
                     size_t n_chebyshev)
{
	if (n_chebyshev <= 1)
		throw std::runtime_error("Need at least 2 Chebyshev points.");

	/* Get the maximum power: */
	double PHmax = -std::numeric_limits<double>::infinity();
	for (size_t i=0; i<N.size(); ++i){
		double PHmax_i = std::numeric_limits<double>::infinity();
		for (size_t j=0; j<N[i]; ++j){
			if (qi[i][j] <= 0)
				throw std::runtime_error("At least one qi is zero or negative "
				                         "and has hence left the model "
				                         "definition space.");
			const double PH = qi[i][j] / ci[i][j];
			if (PH > 0 && PH < PHmax_i){
				PHmax_i = PH;
			}
		}
		if (PHmax_i > PHmax)
			PHmax = PHmax_i;
	}
	if (std::isinf(PHmax))
		throw std::runtime_error("Found infinite maximum power in "
			"posterior_tail_quantiles_batch_barycentric_lagrange."
		);

	/* The support for the interpolation: */
	struct xf_t {
		double x;
		long double f;
	};
	std::vector<xf_t> support(n_chebyshev, xf_t({0.0, 0.0}));
	{
		/* Prepare the interpolation points: */
		std::vector<double> x(n_chebyshev);
		for (size_t i=0; i<n_chebyshev; ++i){
			constexpr long double pi = std::numbers::pi_v<long double>;
			long double z = std::cos(i * pi / (n_chebyshev-1));
			x[i] = std::min(std::max<double>(0.5 * (1.0+z) * PHmax, 0.0),
			                PHmax);
		}

		/* Evaluate the posterior tail: */
		std::vector<long double> f(qi.size() * n_chebyshev);
		posterior_tail_batch(x.data(), n_chebyshev, f.data(), qi, ci, N, p, s,
		                     n, nu, amin, dest_tol, precision);

		/* Transfer to the support vector: */
		for (size_t i=0; i<n_chebyshev; ++i){
			support[i].x = x[i];
		}
		for (size_t j=0; j<qi.size(); ++j){
			for (size_t i=0; i<n_chebyshev; ++i){
				support[i].f += f[n_chebyshev * j + i];
			}
		}
		for (size_t i=0; i<n_chebyshev; ++i){
			support[i].f /= qi.size();
		}
	}

	/* Now we can interpolate: */
	auto tail_bli = [&](double PH) -> double {
		auto xfit = support.cbegin();
		if (PH == xfit->x)
			return xfit->f;
		long double nom = 0.0;
		long double denom = 0.0;
		long double wi = 0.5 / (PH - xfit->x);
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
		return std::max<double>(std::min<double>(nom / denom, 1.0), 0.0);
	};

	/* Now solve for the quantiles: */
	for (size_t i=0; i<Nquant; ++i){
		if (quantiles[i] == 0.0)
			res[i] = PHmax;
		else if (quantiles[i] == 1.0)
			res[i] = 0.0;
		else if (quantiles[i] > 0.0 && quantiles[i] < 1.0){
			/* The typical case. Use TOMS 748 on a quantile
			 * function to find the quantile.
			 */
			auto quantile_function = [&](double PH) -> double {
				return tail_bli(PH) - quantiles[i];
			};
			std::uintmax_t max_iter(100);
			eps_tolerance<double>
			   eps_tol(std::numeric_limits<double>::digits - 2);
			std::pair<double,double> bracket
			   = toms748_solve(quantile_function, 0.0, PHmax, 1.0-quantiles[i],
			                   -quantiles[i], eps_tol, max_iter);
			res[i] = 0.5*(bracket.first + bracket.second);
		} else {
			throw std::runtime_error("Encountered quantile out of bounds "
			                         "[0,1].");
		}
	}
}


/*
 *
 *   Tail quantiles.
 *
 */


/* Same as the above but capturing the exception and returning an exit
 * code instead (0 if succeeded, 1 if error) */
int tail_quantiles_intcode(const double* quantiles, double* res,
                           const size_t Nquant, const double* qi,
                           const double* ci, const size_t N, const double p,
                           const double s, const double n, const double nu,
                           const double amin, const double dest_tol,
                           short print)
{
	try {
		tail_quantiles(quantiles, res, Nquant, qi, ci, N, p, s, n, nu,
		               amin, dest_tol);
	}  catch (ScaleError<double>& e){
		if (print){
			std::cerr << "ScaleError (" << e.what() << "\n"
			             "log_scale: " << e.log_scale() << "\n";
		}
		return 1;
	} catch (PrecisionError<double>& pe) {
		if (print){
			std::cerr << "A precision error happened.\n" << pe.what() << "\n";
		}
		return 2;
	} catch (std::runtime_error& e){
		if (print){
			std::cerr << "A runtime error happened.\n" << e.what() << "\n";
		}
		return 3;
	} catch (std::exception& e) {
		if (print){
			std::cerr << "Another error happened.\n" << e.what() << "\n";
		}
		return 4;
	}
	// All went well.
	return 0;
}



int check_locals(const double* qi, const double* ci, size_t N, double p0,
                 double s0, double n0, double v0, double amin, double dest_tol,
                 double lp, double ls, double n, double v, double Qmax,
                 double h0, double h1, double h2, double h3, double w,
                 double lh0, double l1p_w)
{
	locals_t<long double> L;
	size_t imax;
	init_locals(qi, ci, N, (long double)v0, (long double)n0, (long double)s0,
	            (long double)p0, (long double)amin,
	            nullptr, 0, L, imax, dest_tol);

	auto compare = [](double varcmp, double var0, const char* name) -> int
	{
		if (std::fabs(varcmp - var0) > 1e-3 * std::fabs(var0)){
			std::cout << "\e[31mVariable '" << name << "' difference: "
			          << varcmp << " vs " << var0 << ".\e[0m\n" << std::flush;
			return 1;
		} else {
			std::cout << "Variable '" << name << "' ok.\n" << std::flush;
			return 0;
		}
	};

	int code = 0;
	code += compare(lp, L.lp, "lp");
	code += compare(ls, L.ls, "ls");
	code += compare(n,  L.n, "n");
	code += compare(v,  L.v, "v");
	code += compare(Qmax, L.Qmax, "Qmax");
	code += compare(h0, L.h0, "h0");
	code += compare(h1, L.h1, "h1");
	code += compare(h2, L.h2, "h2");
	code += compare(h3, L.h3, "h3");
	code += compare(w, L.w, "w");
	code += compare(lh0, L.lh0, "lh0");
	code += compare(l1p_w, L.l1p_w, "l1p_w");

	return code;
}




}
}
