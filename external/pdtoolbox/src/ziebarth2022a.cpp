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
/*
 * Use a high precision floating point format.
 */
#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
using boost::multiprecision::float128;
typedef float128 real_t;
#else
typedef long double float128;
typedef long double real_t;
#endif


/* tanh_sinh tolerance: */
constexpr double TOL_TANH_SINH
     = cnst_sqrt(std::numeric_limits<double>::epsilon());


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


/*
 * Get a scale factor for the integrals by computing an approximate maximum
 * of the integrand of the alpha integral (disregarding the z-integral, i.e.
 * assuming constant z):
 */

template<typename real>
real log_integrand_amax(const real v, const real lp, const real n,
                        const real ls, const real l1pwz, const real lkiz_sum)
{
	using std::abs;
	using boost::multiprecision::abs;

	/* Uses Newton-Raphson to compute the (approximate) maximum of the
	 * integrand (disregarding the second integral) of the normalization
	 * of the posterior.
	 */
	const real C = lp - v*(ls + l1pwz) + lkiz_sum;
	real a = 1.0;
	real f0, f1, da;

	for (size_t i=0; i<20; ++i){
		f0 = v * digamma(v*a) - n * digamma(a) + C;
		f1 = v*v*trigamma(v*a) - n*trigamma(a);
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
itgmax_t<real> log_integrand_maximum(const real v, const real lp, const real n,
                                     const real ls, const real l1pwz,
                                     const real lkiz_sum)
{
	using std::lgamma;
	using boost::multiprecision::lgamma;

	itgmax_t<real> res;
	res.a = log_integrand_amax(v, lp, n, ls, l1pwz, lkiz_sum);

	res.logI = lgamma(v*res.a) + (res.a-1.0) * lp - n*lgamma(res.a)
	           - v*res.a*(ls + l1pwz) + (res.a-1) * lkiz_sum;

	return res;
}


/*
 * The innermost integrand of the double integral; the integrand in `a`.
 */
template<bool log_integrand, typename real>
real inner_integrand_template(const real a, const real lp,
                              const real ls, const real n,
                              const real v, const real l1p_kiz_sum,
                              const real l1p_wz, const real log_scale
                             )
{
	using std::exp;
	using boost::multiprecision::exp;
	using std::lgamma;
	using boost::multiprecision::lgamma;
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::isnan;
	using boost::multiprecision::isnan;

	auto va = v * a;

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
	real lS = (a-1) * l1p_kiz_sum;

	// Term ( 1 / (s_new * (1-w*z))) ^ va
	lS -= va * (ls + l1p_wz);

	// Remaining summands:
	lS += lgva + (a-1.0) * lp - n * lga - log_scale;

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
real outer_integrand(const real z, const real lp, const real ls, const real n,
                     const real v, const std::vector<real>& ki,
                     const real w, const real log_scale, const real amin,
                     const real Iref)
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

	if (isnan(z))
		throw std::runtime_error("`z` is nan!");

	// Set the inner parameters:
	real l1p_kiz_sum = 0.0;
	for (const real& k : ki)
		l1p_kiz_sum += log1p(-k * z);
	const real l1p_wz = log1p(-w * z);

	/* The non-degenerate case.
	 * First compute the maximum of the a integrand at this given z:
	 */
	const itgmax_t<real> lImax = log_integrand_maximum(v, lp, n, ls, l1p_wz,
	                                                   l1p_kiz_sum);

	// Integrate:
	auto integrand = [&](real a) -> real {
		real result = inner_integrand_template<false>(a, lp, ls, n, v,
		                                              l1p_kiz_sum, l1p_wz,
		                                              lImax.logI);
		return result;
	};

	real error, L1, S;
	try {
		/* We wrap the integration into a try-catch to be able to distinguish
		 * the source of the error if any should be thrown.
		 */
		if (lImax.a > amin){
			size_t lvl0, lvl1;
			real error1, L11;
			tanh_sinh<real> int0;
			exp_sinh<real> int1;
			S = int0.integrate(integrand, amin, lImax.a,
			                   TOL_TANH_SINH, &error, &L1, &lvl0)
			  + int1.integrate(integrand, lImax.a,
			                   std::numeric_limits<real>::infinity(),
			                   TOL_TANH_SINH, &error1, &L11, &lvl1);
			L1 += L1;
			error += error1;
		} else {
			size_t levels;
			tanh_sinh<real> integrator;
			S = integrator.integrate(integrand, amin,
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
		throw ScaleError<real>("outer_integrand", log_scale);
	}

	/* Error checking: */
	constexpr double ltol = std::log(1e-5);
	const real cmp_err = std::max<real>(log(L1),
	                                    log(Iref) + log_scale - lImax.logI);
	if (log(error) > ltol + cmp_err){
		/* Check if this is relevant: */
		throw PrecisionError<real>("outer_integrand", error, L1);
	}

	return S * exp(lImax.logI - log_scale);
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
az_t<real> log_integrand_max(const real lp, const real ls, const real n,
                             const real v, const real w, const real amin,
                             const std::vector<real>& ki)
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
		for (const real& k : ki)
			l1p_kiz_sum += log1p(-k * z);
		real l1p_wz = log1p(-w * z);

		/* First and second derivatives of the above by z: */
		real k_1mkz_sum = 0.0;
		real k2_1mkz2_sum = 0.0;
		for (real k : ki){
			real x = k / (1.0 - k*z);
			k_1mkz_sum -= x;
			k2_1mkz2_sum -= x*x;
		}
		real w_1mwz = -w / (1.0 - w*z);
		real w2_1mwz2 = - w_1mwz * w_1mwz;


		/* New amax: */
		real amax = std::max(log_integrand_amax(v, lp, n, ls, l1p_wz,
		                                        l1p_kiz_sum),
		                     amin);

		/* Log of integrand:
		 * f0 =   std::lgamma(v*amax) + (amax - 1.0) * lp
		 *           - n*std::lgamma(amax) - v*amax*(ls + l1p_wz)
		 *           + (amax - 1.0) * l1p_kiz_sum
		 */

		/* Derivative of the log of the integrand by z: */
		real f1 = - v * amax * w_1mwz + (amax - 1.0) * k_1mkz_sum;

		/* Second derivative of the log of the integrand by z: */
		real f2 = - v * amax * w2_1mwz2 + (amax - 1.0) * k2_1mkz2_sum;

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
	for (const real& k : ki)
		l1p_kiz_sum += log1p(-k * z);
	real l1p_wz = log1p(-w * z);
	real amax = std::max(log_integrand_amax(v, lp, n, ls, l1p_wz, l1p_kiz_sum),
	                     amin);

	/* Log of the integrand: */
	const real f0 = lgamma(v*amax) + (amax - 1.0) * lp
	                - n*lgamma(amax) - v*amax*(ls + l1p_wz)
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

	C1_t(const real a, const real v, const real w, const real h0,
	     const real h1)
	{
		auto h1h0 = h1 / h0;
		deriv0 = (h1h0 + v*w/(w-1))*a - h1h0;
		deriv1 = h1h0 + v*w/(w-1);
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

	C2_t(const real a, const real v, const real w, const real h0,
	     const real h1, const real h2)
	{
		auto h1h0 = h1 / h0;
		auto h2h0 = h2 / h0;
		auto nrm = 1.0 / (w*w - 2*w + 1);
		auto D2 = (v*v*w*w + h1h0 * (2 * v * w * (w - 1)
		                             +  h1h0 * (1 + w * (w - 2)))
		          ) * nrm;
		auto D1 = (v*w*w + 2*h2h0*(w*(w - 2) + 1)
		           - h1h0 * (2*w*v*(w - 1) + 3 * h1h0*(w*(w-2)+1))) * nrm;
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

	C3_t(const real a, const real v, const real w,
	     const real h0, const real h1, const real h2,
	     const real h3)
	{
		auto h1h0 = h1 / h0;
		auto h2h0 = h2 / h0;
		auto h3h0 = h3 / h0;
		auto v2 = v*v;
		auto v3 = v2*v;
		auto w2 = w*w;
		auto w3 = w2*w;
		auto F = w * (w * (w - 3) + 3) - 1; // w3 - 3*w2 + 3*w - 1
		auto nrm = 1/F;
		auto D3 = (v3*w3
		           + h1h0 * (3 * v2 * w2 * (w-1)
		                     + h1h0 * (3 * v*w*(w*(w-2)+1)
		                               + h1h0*F))
		          ) * nrm;

		auto D2 = 3 * (v2*w3 + 2 * h2h0 * v * w * (w-1) * (w-1)
		               + h1h0 * (v * w2 * (v-1) * (1-w)
		                         + 2 * h2h0 * F
		                         + h1h0 * (3 * v * w * (w * (2-w) - 1)
		                                   - 2 * h1h0 * F))
		            ) * nrm;
		auto D1 = (  2 * v * w3  +  6*h3h0 * F
		           + 6 * h2h0 * v * w * (w * (2 - w) - 1)
		           + h1h0 * (3 * v * w2 * (1-w)
		                     - 18 * h2h0 * F
		                     + h1h0 * (6 * v * w * (w2 - 2*w + 1)
		                               + 11 * h1h0*F))
		          ) * nrm;

		auto D0 = 6 * (h1h0 * (2 * h2h0 - h1h0*h1h0) - h3h0);

		deriv0 = D0 + a * (D1 + a * (D2 + a * D3));
		deriv1 = D1 + a * (2*D2 + a * 3 * D3);
		deriv2 = 2 * (D2 + 3 * a * D3);
	}
};


template<typename real, unsigned char order>
real large_z_amax(const real lp, const real ls, const real n,
                  const real v, const real h0, const real h1,
                  const real h2, const real h3, const real w,
                  const real l1pw, const real ym, const real amin)
{
	using std::log;
	using boost::multiprecision::log;
	using std::abs;
	using boost::multiprecision::abs;

	/* This method computes max location of the maximum of the four integrals
	 * used in the Taylor expansion of the double integral for large z. */
	constexpr double TOL = 1e-14;
	const real lh0 = log(h0);
	const real lym =  log(ym);
	real amax = (n > v) ? std::max<real>((lp - v*ls + v*log(v) + lh0 - v * l1pw
	                                      + lym) / (n - v), 1.0)
	                    : 1.0;

	/* Recurring terms of the first and second derivatives of the a-integrands:
	 */
	auto f0_base = [&](real a) -> real {
		return v * digamma(v*a) + lp - n*digamma(a) - v*ls + lh0 - v*l1pw;
	};
	auto f1_base = [&](real a) -> real {
		return v*v*trigamma(v*a) - n*trigamma(a);
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
			C1_t<real> C1(amax, v, w, h0, h1);
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
			C2_t<real> C2(amax, v, w, h0, h1, h2);
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
			C3_t<real> C3(amax, v, w, h0, h1, h2, h3);
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
	return std::max(amax, amin);
}



struct params3_t {
	// Additional parameters:
	double lh0;
	double h0;
	double h1;
	double h2;
	double h3;
	bool y_integrated;

	// Parameters:
	double nu_new;
	double lp_tilde;
	double n_new;
	double ls_tilde;
	double l1p_w;
	double w;

	params3_t(){};

	params3_t(double lh0, double h0, double h1, double h2, double h3,
	          bool y_integrated, double nu_new, double lp_tilde, double n_new,
	          double ls_tilde, double l1p_w, double w)
	   : lh0(lh0), h0(h0), h1(h1), h2(h2), h3(h3), y_integrated(y_integrated),
	     nu_new(nu_new), lp_tilde(lp_tilde), n_new(n_new), ls_tilde(ls_tilde),
	     l1p_w(l1p_w), w(w)
	{};
};

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
a_integral_large_z_log_integrand(real a, real ly, real log_scale,
                                 const real lp, const real ls, const real n,
                                 const real v, const real w, const real h0,
                                 const real h1, const real h2, const real h3,
                                 const real l1p_w, const real lh0)
{
	using std::log;
	using boost::multiprecision::log;
	using std::lgamma;
	using boost::multiprecision::lgamma;
	using std::isinf;
	using boost::multiprecision::isinf;

	const real va = v * a;

	// Compute C:
	real lC = 0.0;
	int8_t sign = 1;
	if (C == C_t::C0){
		/* C0 = 1 */
		lC = 0.0;
	} else if (C == C_t::C1) {
		/* C1 */
		const C1_t C1(a, v, w, h0, h1);
		if (C1.deriv0 < 0){
			sign = -1;
			lC = log(-C1.deriv0);
		} else {
			lC = log(C1.deriv0);
		}
	} else if (C == C_t::C2){
		/* C2 */
		const C2_t C2(a, v, w, h0, h1, h2);
		if (C2.deriv0 < 0){
			sign = -1;
			lC = log(-C2.deriv0);
		} else {
			lC = log(C2.deriv0);
		}
	} else if (C == C_t::C3){
		/* C3 */
		const C3_t C3(a, v, w, h0, h1, h2, h3);
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
	lC -= va * (ls + l1p_w);

	// Remaining summands:
	lC += lgva + (a-1.0) * (lp + lh0) - n * lga - log_scale;

	if (isinf(lC)){
		return {.log_abs=-std::numeric_limits<double>::infinity(),
		        .sign=sign};
	}

	return {.log_abs=lC, .sign=sign};
}

template<C_t C, bool y_integrated, typename real>
real a_integral_large_z_integrand(real a, real ly, real log_scale,
                                  const real lp, const real ls, const real n,
                                  const real v, const real w, const real h0,
                                  const real h1, const real h2, const real h3,
                                  const real l1p_w, const real lh0)
{
	using std::exp;
	using boost::multiprecision::exp;
	using std::isinf;
	using boost::multiprecision::isinf;

	log_double_t<real> res
	    = a_integral_large_z_log_integrand<C,y_integrated, real>(a, ly,
	                        log_scale, lp, ls, n, v, w, h0, h1, h2, h3, l1p_w,
	                        lh0);

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
real y_taylor_transition_root_backend(real y, const real lp, const real ls,
                            const real n, const real v, const real h0,
                            const real h1, const real h2, const real h3,
                            const real lh0, const real w, const real l1p_w,
                            const real log_scale0, const real amin)
{
	using std::log;
	using boost::multiprecision::log;
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::abs;
	using boost::multiprecision::abs;

	constexpr double epsilon = 1e-14;

	/* Get the scale: */
	const real amax = large_z_amax<real,0>(lp, ls, n, v, h0, h1, h2, h3,
	                                       w, l1p_w, y, amin);
	const real ly = log(y);
	const real log_scale = log_scale0
	    + a_integral_large_z_log_integrand<C_t::C0,true>(amax, ly, log_scale0,
	                                                lp,  ls, n, v, w, h0, h1,
	                                                h2, h3, l1p_w,
	                                                lh0).log_abs;

	/* Compute the 'a' integrals for the constant and the cubic term: */
	real error, L1;
	size_t levels;
	exp_sinh<real> es_integrator;
	tanh_sinh<real> ts_integrator;

	auto integrand0 = [&](real a) -> real
	{
		return a_integral_large_z_integrand<C_t::C0,true,real>(
		                a, ly, log_scale, lp, ls, n, v, w, h0, h1,
		                h2, h3, l1p_w, lh0
		);
	};

	real S0;
	if (amax > amin){
		real error1, L11;
		S0 = ts_integrator.integrate(integrand0, amin, amax,
		                             TOL_TANH_SINH, &error, &L1, &levels)
		   + es_integrator.integrate(integrand0, amax,
		                             std::numeric_limits<real>::infinity(),
		                             TOL_TANH_SINH, &error1, &L11, &levels);
		error += error1;
		L1 += L11;
	} else {
		S0 = es_integrator.integrate(integrand0, amin,
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


	auto integrand1 = [&](real a) -> real
	{
		return a_integral_large_z_integrand<C_t::C3,true,real>(
		                a, ly, log_scale, lp, ls, n, v, w, h0, h1,
		                h2, h3, l1p_w, lh0
		);
	};


	real S1;
	if (amax > amin){
		real error1, L11;
		S1 = ts_integrator.integrate(integrand1, amin, amax,
		                             TOL_TANH_SINH, &error, &L1, &levels)
		   + es_integrator.integrate(integrand1, amax,
		                             std::numeric_limits<real>::infinity(),
		                             TOL_TANH_SINH, &error1, &L11, &levels);
		error += error1;
		L1 += L11;
	} else {
		S1 = es_integrator.integrate(integrand1, amin,
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
real y_taylor_transition(const real h0, const real h1, const real h2,
                         const real h3, const real nu_new,
                         const real lp_tilde, const real n_new,
                         const real log_scale, const real ls_tilde,
                         const real l1p_w, const real w, const real amin,
                         const real ymin = 1e-32)
{
	using std::log;
	using boost::multiprecision::log;
	using std::isnan;
	using boost::multiprecision::isnan;

	/* Find a value above the threshold: */
	real yr = 1e-9;
	real lh0 = log(h0);
	real val = y_taylor_transition_root_backend<real>(yr, lp_tilde, ls_tilde,
	                                                  n_new, nu_new, h0, h1, h2,
	                                                  h3, lh0, w, l1p_w,
	                                                  log_scale, amin);
	while (val < 0 || isnan(val)){
		yr = std::min<real>(2*yr, 1.0);
		if (yr == 1.0)
			break;
		val = y_taylor_transition_root_backend<real>(yr, lp_tilde, ls_tilde,
		                                       n_new, nu_new, h0, h1, h2, h3,
		                                       lh0, w, l1p_w, log_scale, amin);
	}

	/* Root finding: */
	auto rootfun = [&](real y) -> real {
		return y_taylor_transition_root_backend<real>(y, lp_tilde, ls_tilde,
		                                        n_new, nu_new, h0, h1, h2, h3,
		                                        lh0, w, l1p_w, log_scale,
		                                        amin);
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
real a_integral_large_z(const real ym, const real h0, const real h1,
                        const real h2, const real h3, const real nu_new,
                        const real lp_tilde, const real n_new,
                        const real log_scale, const real ls_tilde,
                        const real l1p_w, const real w, const real amin,
                        const real S_cmp)
{
	using std::log;
	using boost::multiprecision::log;
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::abs;
	using boost::multiprecision::abs;
	using std::isnan;
	using boost::multiprecision::isnan;

	/* Set the integrand's non-varying parameters: */
	const real ly = log(ym);
	const real lh0 = log(h0);

	/* Integration setup for 'a' integrals:: */
	real error, L1, S;
	size_t levels;
	exp_sinh<real> integrator;

	auto integrand = [&](real a) -> real
	{
		real S0 = a_integral_large_z_integrand<C_t::C0, y_integrated,real>(
		               a, ly, log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		               w, h0, h1, h2, h3, l1p_w, lh0
		);
		real S1 = a_integral_large_z_integrand<C_t::C1,y_integrated,real>(
		               a, ly,log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		               w, h0, h1, h2, h3, l1p_w, lh0
		);
		real S2 = a_integral_large_z_integrand<C_t::C2,y_integrated,real>(
		               a, ly,log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		               w, h0, h1, h2, h3, l1p_w, lh0
		);
		real S3 = a_integral_large_z_integrand<C_t::C3,y_integrated,real>(
		              a, ly, log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		              w, h0, h1, h2, h3, l1p_w, lh0
		);
		return S0 + S1 + S2 + S3;
	};

	try {
		S = integrator.integrate(integrand, amin,
		                         std::numeric_limits<real>::infinity(),
		                         TOL_TANH_SINH, &error, &L1, &levels);
	} catch (std::runtime_error& e) {
		std::string msg("Error in a_integral_large_z exp_sinh routine: '");
		msg += e.what();
		msg += "'.";
		throw std::runtime_error(msg);
	}
	if (isinf(S) || isnan(S))
		throw ScaleError<real>("a_integral_large_z", log_scale);
	if (error > 1e-14 * std::max(L1, S_cmp))
		throw PrecisionError<real>("a_integral_large_z_S3", error, L1);

	return S;
}




/*
 * Compute the posterior:
 */

using pdtoolbox::heatflow::posterior_t;


template<typename real, bool need_norm=true>
void init_locals(const double* qi, const double* ci, size_t N,
                 const real nu, const real n, const real s, const real p,
                 const real amin,
                 // Destination parameters:
                 real& lp_tilde, real& ls_tilde, real& nu_new,
                 real& n_new, real& Qmax, real& h0, real& h1,
                 real& h2, real& h3, real& w, real& l1p_w,
                 real& ztrans, real& log_scale, real& norm,
                 real& Iref, std::vector<real>& ki, real& full_taylor_integral,
                 size_t& imax)
{
	using std::isinf;
	using boost::multiprecision::isinf;
	using std::log;
	using boost::multiprecision::log;
	using std::log1p;
	using boost::multiprecision::log1p;

	// Initialize parameters:
	nu_new = nu + N;
	n_new = n + N;

	// Step 0: Determine Qmax, A, B, and ki:
	imax = 0;
	Qmax = std::numeric_limits<real>::infinity();
	for (size_t i=0; i<N; ++i){
		if (qi[i] <= 0)
			throw std::runtime_error("At least one qi is zero or negative and "
			                         "has hence left the model definition "
			                         "space.");
		real Q = qi[i] / ci[i];
		if (Q > 0 && Q < Qmax){
			Qmax = Q;
			imax = i;
		}
	}
	if (isinf(Qmax))
		throw std::runtime_error("Found Qmax == inf. The model is not "
		                         "well-defined. This might happen if all "
		                         "ci <= 0, i.e. heat flow anomaly has a "
		                         "negative or no impact on all data points.");

	real A = s;
	for (size_t i=0; i<N; ++i)
		A += qi[i];
	ls_tilde = log(A);
	real B = 0.0;
	for (size_t i=0; i<N; ++i)
		B += ci[i];
	for (size_t i=0; i<N; ++i)
		ki[i] = ci[i] * Qmax / qi[i];
	real lq_sum = 0.0;
	for (size_t i=0; i<N; ++i)
		lq_sum += log(qi[i]);

	lp_tilde = log(p) + lq_sum;
	w = B * Qmax / A;

	// Integration config:

	/* Get an estimate of the scaling: */

	/* The new version: */
	az_t azmax = log_integrand_max(lp_tilde, ls_tilde, n_new, nu_new, w, amin,
	                               ki);
	log_scale = azmax.log_integrand;

	/* Compute the coefficients for the large-z (small-y) Taylor expansion: */
	h0=1.0;
	h1=0;
	h2=0;
	h3=0;
	for (size_t i=0; i<N; ++i){
		if (i == imax)
			continue;
		const real d0 = 1.0 - ki[i];
		h3 = d0 * h3 + ki[i] * h2;
		h2 = d0 * h2 + ki[i] * h1;
		h1 = d0 * h1 + ki[i] * h0;
		h0 *= d0;
	}

	/* Step 1: Compute the normalization constant.
	 *         This requires the full integral and might require
	 *         a readjustment of the log_scale parameter. */
	l1p_w = log1p(-w);
	bool norm_success = false;
	full_taylor_integral = std::nan("");
	norm = std::nan("");
	ztrans = std::nan("");
	real ymax, S;

	while (!norm_success && !isinf(log_scale)){
		// Compute the transition z where we switch to an analytic integral of
		// the Taylor expansion:
		try {
			ymax = y_taylor_transition(h0, h1, h2, h3, nu_new, lp_tilde, n_new,
			                           log_scale, ls_tilde, l1p_w, w,
			                           static_cast<real>(1e-32));
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
			log_scale += s.log_scale();
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
		ztrans = 1.0 - ymax;

		/*
		 * For the unnormed posterior, we do not need the normalization
		 * constant.
		 */
		if (!need_norm)
			return;

		try {
			/* Compute a reference integral along the a axis at z that
			 * corresponds to the maximum (a,z): */
			Iref = 0;
			Iref = outer_integrand(azmax.z, lp_tilde, ls_tilde, n_new,
			                       nu_new, ki, w, log_scale, amin, Iref);

			auto integrand = [&](real z) -> real
			{
				return outer_integrand(z, lp_tilde, ls_tilde, n_new, nu_new,
				                       ki, w, log_scale, amin, Iref);
			};

			// 1.1: Double numerical integration in z range [0,1-ymax]:
			static_assert(std::is_same<real, double>::value ||
			              std::is_same<real, long double>::value ||
			              std::is_same<real, float>::value ||
			              std::is_same<real, float128>::value);
			real error;
			tanh_sinh<real> integrator;
			S = integrator.integrate(integrand, static_cast<real>(0), ztrans,
			                         TOL_TANH_SINH, &error);

		} catch (ScaleError<real>& s) {
			if (s.log_scale() < 0)
				throw std::runtime_error("Failed to determine the "
				                         "normalization constant: Encountered "
				                         "negative scale error.");
			log_scale += s.log_scale();
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
			msg.append(std::to_string(static_cast<double>(ztrans)));
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
			full_taylor_integral =
			a_integral_large_z<true>(ymax, h0, h1, h2, h3, nu_new, lp_tilde,
			                   n_new, log_scale, ls_tilde, l1p_w, w, amin,
			                   S);
			S += full_taylor_integral;
		} catch (ScaleError<real>& s) {
			if (s.log_scale() < 0)
				throw std::runtime_error("Failed to determine the "
				                         "normalization constant: Encountered "
				                         "negative scale error.");
			log_scale += s.log_scale();
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
			msg.append(std::to_string(static_cast<double>(ztrans)));
			msg.append(", ymax=");
			std::ostringstream os_ymax;
			os_ymax << ymax;
			msg.append(os_ymax.str());
			throw std::runtime_error(msg);
		}
		norm = S;
		norm_success = true;
	}
}



/*
 *
 *   Computing the posterior.
 *
 */

template<posterior_t type>
void posterior(const double* x, double* res, size_t Nx, const double* qi,
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

	//typedef long double real;
	//typedef boost::multiprecision::cpp_dec_float_100 real;
	typedef real_t real;

	typedef gauss_kronrod<real, 15> GK;


	/* Step 1: Use the common parameter initialization routine which
	 *         mainly determines the normalization constant: */
	real lp_tilde=0, ls_tilde=0, nu_new=0, n_new=0, Qmax=0, h0=0, h1=0,
	     h2=0, h3=0, w=0, l1p_w=0, ztrans=0, log_scale=0, norm=0,
	     full_taylor_integral=0, Iref=0;
	size_t imax;
	std::vector<real> ki(N);
	if (type == posterior_t::UNNORMED_LOG)
		init_locals<real,false>(qi, ci, N, (real)nu, (real)n, (real)s, (real)p,
		            (real)amin, lp_tilde, ls_tilde, nu_new, n_new, Qmax, h0, h1,
		            h2, h3, w, l1p_w, ztrans, log_scale, norm, Iref, ki,
		            full_taylor_integral, imax);
	else
		init_locals(qi, ci, N, (real)nu, (real)n, (real)s, (real)p, (real)amin,
		            lp_tilde, ls_tilde, nu_new, n_new, Qmax, h0, h1, h2, h3, w,
		            l1p_w, ztrans, log_scale, norm, Iref, ki,
		            full_taylor_integral, imax);

	std::vector<size_t> order(0);
	bool is_cumulative = type & (posterior_t::CUMULATIVE | posterior_t::TAIL);
	real z_last = 0.0, S_cumul = 0.0;
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
			/* We start cumulatively from the front: */
			z_last = 0.0;
		} else {
			std::sort(order.begin(), order.end(),
			          [&](size_t i, size_t j) -> bool
			          {
			              return x[i] > x[j];
			          }
			);
			/* We start cumulatively from the back: */
			z_last = 1.0;
		}
	}

	auto integrand = [&](real z) -> real {
		return outer_integrand<real>(z, lp_tilde, ls_tilde, n_new, nu_new, ki,
		                             w, log_scale, amin, Iref);
	};

	/* A condition for deciding the integrator: */
	bool use_thsh = false;
	if (type == posterior_t::TAIL || type == posterior_t::CUMULATIVE)
		use_thsh = (integrand(ztrans)*ztrans > 2*norm);

	tanh_sinh<real> thsh_int;
	// /* Step 2: Compute the values: */
	for (size_t i=0; i<Nx; ++i){
		const size_t j = (is_cumulative) ? order[i] : i;
		// Short cut: x out of bounds:
		if (x[j] < 0){
			if (type == posterior_t::UNNORMED_LOG)
				res[j] = -std::numeric_limits<double>::infinity();
			else if (type == posterior_t::TAIL)
				res[j] = 1.0;
			else
				res[j] = 0.0;
			continue;
		} else if (x[j] >= Qmax){
			if (type == posterior_t::DENSITY || type == posterior_t::TAIL)
				res[j] = 0.0;
			else if (type == posterior_t::UNNORMED_LOG)
				res[j] = -std::numeric_limits<double>::infinity();
			else
				res[j] = 1.0;
			continue;
		}

		/* Continue depending on chosen representation of the posterior: */
		const real zi = x[j] / Qmax;
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
			if (zi <= ztrans)
				res[j] = static_cast<double>(integrand(zi) / (Qmax * norm));
			else
				// Use the Taylor expansion for the limit z->1, y=1-z -> 0:
				res[j] = static_cast<double>(
				             a_integral_large_z<false,real>(1.0-zi, h0, h1, h2,
				                       h3, nu_new, lp_tilde, n_new, log_scale,
				                       ls_tilde, l1p_w, w, amin, norm)
				         / (Qmax * norm)
				);
		} else if (type == posterior_t::CUMULATIVE){
			real S;
			if (zi <= ztrans){
				real err;
				const real eps_tol
				   = boost::math::tools::root_epsilon<double>();
				if (z_last == zi){
					S = 0;
				} else {
					S = GK::integrate(integrand, z_last, zi, 1,
					                  eps_tol, &err);
					if (err > 1e-2 * dest_tol * norm) {
						if (S != 0)
							S = GK::integrate(integrand, z_last, zi, 15,
								              1e-2 * dest_tol * norm / S);
						else
							S = GK::integrate(integrand, z_last, zi, 15);
					}
				}
				S_cumul += S;
				z_last = zi;

				/* Sanity check: Crosscheck the total integral against
				 * the Gauss-Kronrod-derived norm: */
				if (i+1 == Nx){
					S = GK::integrate(integrand, z_last,
					                                ztrans);
					real norm_num = S + S_cumul + full_taylor_integral;
					if (abs(norm_num - norm) > dest_tol * norm){
						throw PrecisionError<real>("CDF normalization failed "
						                     "to achieve the desired "
						                     "precision.",
						                     abs(norm_num - norm), norm);
					}
					/* Renormalize: */
					const double rescale = 1.0 / static_cast<double>(norm_num);
					for (size_t k=0; k<Nx; ++k)
						res[k] *= rescale;
				}
			} else {
				/* Sanity check: */
				if (z_last < ztrans){
					S = GK::integrate(integrand, z_last, ztrans);
					real norm_num = S + S_cumul + full_taylor_integral;
					if (abs(norm_num - norm) > dest_tol * norm){
						throw PrecisionError<real>("CDF normalization failed "
						                           "to achieve the desired "
						                           "precision.",
						                           abs(norm_num - norm),
						                           norm);
					}
				}
				/* Now integrate from the back: */
				S = norm
				    - a_integral_large_z<true, real>(1.0-zi, h0, h1, h2, h3,
				                       nu_new, lp_tilde, n_new, log_scale,
				                       ls_tilde, l1p_w, w, amin, norm);
				S_cumul += S;
				z_last = zi;
			}
			res[j] = static_cast<double>(
			    std::max<real>(std::min<real>(S_cumul / norm, 1.0), 0.0)
			);
		} else if (type == posterior_t::TAIL){
			real S;
			if (zi <= ztrans){
				// First the part from zi to ztrans:
				if (z_last >= ztrans){
					z_last = ztrans;
					S_cumul = full_taylor_integral;
				}
				if (use_thsh)
					S = thsh_int.integrate(integrand, zi, z_last);
				else {
					real err;
					constexpr double eps_tol
					   = boost::math::tools::root_epsilon<double>();
					if (zi == z_last){
						S = 0.0;
					} else {
						S = GK::integrate(integrand, zi, z_last, 1,
						                  (real)eps_tol, &err);
						if (err > 1e-2 * dest_tol * norm) {
							if (S != 0)
								S = GK::integrate(integrand, zi, z_last,
								                  15,
								                  1e-2 * dest_tol
								                       * norm / S);
							else
								S = GK::integrate(integrand, zi, z_last);
						}
					}
				}
				S_cumul += S;
				z_last = zi;
			} else {
				// The part from zi to 1:
				S_cumul = a_integral_large_z<true, real>(1.0-zi, h0, h1, h2,
				                       h3, nu_new, lp_tilde, n_new, log_scale,
				                       ls_tilde, l1p_w, w, amin, norm);
				z_last = zi;
			}
			res[j] = static_cast<double>(
			    std::max<real>(std::min<real>(S_cumul / norm, 1.0), 0.0)
			);

			/* Sanity check of the normalization: */
			if (i+1 == Nx && z_last < ztrans){
				if (z_last > 0)
					if (use_thsh)
						S = thsh_int.integrate(integrand, 0, z_last);
					else
						S = GK::integrate(integrand, 0, z_last);
				else
					S = 0.0;
				real norm_num = S + S_cumul;
				if (abs(norm_num - norm) > dest_tol * norm){
					throw PrecisionError<real>("Tail distribution "
					                           "normalization failed to "
					                           "achieve the desired "
					                           "precision.",
					                          abs(norm_num - norm),
					                           norm);
				}
				/* Renormalize: */
				const double rescale = static_cast<double>(norm / norm_num);
				for (size_t k=0; k<Nx; ++k)
					res[k] *= rescale;
			}

		} else if (type == posterior_t::UNNORMED_LOG){
			/* Return the logarithm of the unnormed density.
			 * This mode is useful if external code is used to superpose
			 * additional dimensions that need to be taken into account
			 * when normalizing.
			 */
			if (zi <= ztrans){
				res[j] = static_cast<double>(log(integrand(zi)) + log_scale);
			} else {
				res[j] = static_cast<double>(
				    log(a_integral_large_z<false,real>(1.0-zi, h0, h1,
				             h2, h3, nu_new, lp_tilde, n_new, log_scale,
				             ls_tilde, l1p_w, w, amin, 0.0))
				         + log_scale
				);
			}
		}
	}
}


/*
 * Template instantiations:
 */
namespace pdtoolbox {
namespace heatflow {

void posterior_pdf(const double* x, double* res, size_t Nx, const double* qi,
                   const double* ci, size_t N, double p, double s, double n,
                   double nu, double amin, double dest_tol)
{
	posterior<DENSITY>(x, res, Nx, qi, ci, N, p, s, n, nu, amin, dest_tol);
}


void posterior_pdf_batch(const double* x, size_t Nx, double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double amin, double dest_tol)
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
				              nu, amin, dest_tol);
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


void posterior_cdf(const double* x, double* res, size_t Nx, const double* qi,
                   const double* ci, size_t N, double p, double s, double n,
                   double nu, double amin, double dest_tol)
{
	posterior<CUMULATIVE>(x, res, Nx, qi, ci, N, p, s, n, nu, amin, dest_tol);
}


void posterior_cdf_batch(const double* x, size_t Nx, double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double amin, double dest_tol)
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
				              nu, amin, dest_tol);
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


void posterior_tail(const double* x, double* res, size_t Nx, const double* qi,
                    const double* ci, size_t N, double p, double s, double n,
                    double nu, double amin, double dest_tol)
{
	posterior<TAIL>(x, res, Nx, qi, ci, N, p, s, n, nu, amin, dest_tol);
}


void posterior_tail_batch(const double* x, size_t Nx, double* res,
                          const std::vector<const double*>& qi,
                          const std::vector<const double*>& ci,
                          const std::vector<size_t>& N,
                          double p, double s, double n, double nu,
                          double amin, double dest_tol)
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
				               nu, amin, dest_tol);
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


void posterior_log_unnormed(const double* x, double* res, size_t Nx,
                            const double* qi, const double* ci, size_t N,
                            double p, double s, double n, double nu,
                            double amin, double dest_tol)
{
	posterior<UNNORMED_LOG>(x, res, Nx, qi, ci, N, p, s, n, nu,
	                        amin, dest_tol);
}


/*
 * A version of the above catching errors and returning NaNs:
 */
void posterior_silent(const double* x, double* res, size_t Nx, const double* qi,
               const double* ci, size_t N, double p, double s, double n,
               double nu, double amin, double dest_tol, posterior_t type)
{
	try {
		if (type == DENSITY)
			posterior<DENSITY>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
			                   dest_tol);
		else if (type == CUMULATIVE)
			posterior<CUMULATIVE>(x, res, Nx, qi, ci, N, p, s, n, nu,
			                      amin, dest_tol);
		else if (type == TAIL)
			posterior<TAIL>(x, res, Nx, qi, ci, N, p, s, n, nu, amin,
			                dest_tol);
		else if (type == UNNORMED_LOG)
			posterior<UNNORMED_LOG>(x, res, Nx, qi, ci, N, p, s, n, nu,
			                        amin, dest_tol);
	} catch (std::runtime_error& e) {
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

	/* Step 1: Use the common parameter initialization routine which
	 *         mainly determines the normalization constant: */
	double lp_tilde=0, ls_tilde=0, nu_new=0, n_new=0, Qmax=0, h0=0, h1=0,
	       h2=0, h3=0, w=0, l1p_w=0, ztrans=0, log_scale=0, norm=0,
	       full_taylor_integral=0, Iref=0;
	size_t imax;
	std::vector<double> ki(N);
	init_locals(qi, ci, N, nu, n, s, p, amin, lp_tilde,
	            ls_tilde, nu_new, n_new, Qmax, h0, h1, h2, h3, w, l1p_w,
	            ztrans, log_scale, norm, Iref, ki, full_taylor_integral, imax);


	auto integrand = [&](double z) -> double {
		double result = 0.0;
		if (z <= ztrans)
			result = outer_integrand(z, lp_tilde, ls_tilde, n_new, nu_new,
			                         ki, w, log_scale, amin, Iref) / norm;
		else if (z < 1)
			result = a_integral_large_z<false>(1.0-z, h0, h1, h2, h3,
			                                   nu_new, lp_tilde, n_new,
			                                   log_scale, ls_tilde,
			                                   l1p_w, w, amin, norm) / norm;

		return result;
	};

	size_t i=0;
	try {
		/* Initialize the quantile inverter: */
		QuantileInverter qinv(integrand, 0.0, 1.0, dest_tol, dest_tol, OR);

		/* Now invert: */
		for (; i<Nquant; ++i){
			res[i] = Qmax * qinv.invert(1.0 - quantiles[i]);
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
			res[i] = Qmax * 0.5 * (bracket.first + bracket.second);
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
	struct locals_t {
		double lp_tilde;
		double ls_tilde;
		double nu_new;
		double n_new;
		double Qmax;
		double h0;
		double h1;
		double h2;
		double h3;
		double w;
		double l1p_w;
		double ztrans;
		double log_scale;
		double norm;
		double Iref;
		double full_taylor_integral;
		std::vector<double> ki;
	};
	std::vector<locals_t> locals(qi.size());
	#pragma omp parallel for
	for (size_t i=0; i<qi.size(); ++i){
		if (!error_flag){
			size_t imax;
			try {
				locals[i].ki.resize(N[i]);
				init_locals(qi[i], ci[i], N[i], nu, n, s, p, amin,
				            locals[i].lp_tilde, locals[i].ls_tilde,
				            locals[i].nu_new, locals[i].n_new,
				            locals[i].Qmax, locals[i].h0, locals[i].h1,
				            locals[i].h2, locals[i].h3, locals[i].w,
				            locals[i].l1p_w, locals[i].ztrans,
				            locals[i].log_scale, locals[i].norm,
				            locals[i].Iref, locals[i].ki,
				            locals[i].full_taylor_integral, imax);
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
	double Qmax = 0.0;
	for (size_t i=0; i<qi.size(); ++i){
		Qmax = std::max(Qmax, locals[i].Qmax);

		/* The 'norm' parameter is calculated for z in the range [0,1].
		 * To be able to integrate over the full frictional power space,
		 * need to renorm:
		 */
		locals[i].norm *= locals[i].Qmax;
	}

	auto integrand = [&](double P_H) -> double {
		double result = 0.0;
		for (size_t i=0; i<qi.size(); ++i){
			if (P_H >= locals[i].Qmax)
				continue;
			const double z = P_H / locals[i].Qmax;
			if (z <= locals[i].ztrans)
				result += outer_integrand(z, locals[i].lp_tilde,
				                          locals[i].ls_tilde, locals[i].n_new,
				                          locals[i].nu_new, locals[i].ki,
				                          locals[i].w, locals[i].log_scale,
				                          amin, locals[i].Iref)
				          / locals[i].norm;
			else if (z < 1)
				result += a_integral_large_z<false>(1.0-z, locals[i].h0,
				                           locals[i].h1, locals[i].h2,
				                           locals[i].h3, locals[i].nu_new,
				                           locals[i].lp_tilde, locals[i].n_new,
				                           locals[i].log_scale,
				                           locals[i].ls_tilde, locals[i].l1p_w,
				                           locals[i].w, amin,
				                           locals[i].norm) / locals[i].norm;
		}

		return result / qi.size();
	};

	size_t i=0;
	try {
		/* Initialize the quantile inverter: */
		QuantileInverter qinv(integrand, 0.0, Qmax, dest_tol, dest_tol, OR);

		/* Now invert: */
		for (; i<Nquant; ++i){
			res[i] = qinv.invert(1.0 - quantiles[i]);
		}
	} catch (...) {
		tanh_sinh<double> integrator;
		for (; i<Nquant; ++i){
			auto rootfun = [&](double P_H){
				return integrator.integrate(integrand, P_H, Qmax)
				       - quantiles[i];
			};
			auto tolerance = [&](double a, double b) -> bool {
				return std::fabs(a-b) <= dest_tol;
			};
			std::uintmax_t maxiter = 50;
			std::pair<double,double> bracket
			   = toms748_solve(rootfun, 0.0, Qmax, 1.0-quantiles[i],
			                   -quantiles[i], tolerance, maxiter);
			res[i] = 0.5 * (bracket.first + bracket.second);
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

}
}
