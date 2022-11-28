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

typedef gauss_kronrod<double, 15> GK;

/* A custom exception type indicating that the integral is out of
 * scale for double precision: */
class ScaleError : public std::exception
{
public:
	explicit ScaleError(const char* msg, double log_scale)
	    : _lscale(log_scale), msg(msg) {};

	virtual const char* what() const noexcept
	{
		return msg;
	}

	double log_scale() const
	{
		return _lscale;
	};

private:
	double _lscale;
	const char* msg;
};


class PrecisionError : public std::exception
{
public:
	explicit PrecisionError(const char* msg, double error, double L1)
	    : _error(error), _L1(L1), msg(generate_message(msg, error, L1)) {};

	virtual const char* what() const noexcept
	{
		return msg.c_str();
	}

	double error() const
	{
		return _error;
	};

	double L1() const
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

static double log_integrand_amax(double v, double lp, double n, double ls,
                                 double l1pwz, double lkiz_sum)
{
	/* Uses Newton-Raphson to compute the (approximate) maximum of the
	 * integrand (disregarding the second integral) of the normalization
	 * of the posterior.
	 */
	const double C = lp - v*(ls + l1pwz) + lkiz_sum;
	double a = 1.0;
	double f0, f1, da;

	for (size_t i=0; i<20; ++i){
		f0 = v * digamma(v*a) - n * digamma(a) + C;
		f1 = v*v*trigamma(v*a) - n*trigamma(a);
		da = f0 / f1;
		a -= da;
		a = std::max(a, 1e-8);
		if (std::fabs(da) <= 1e-8 * a)
			break;
	}
	return a;
}

static double log_integrand_maximum(double v, double lp, double n, double ls,
                                    double l1pwz, double lkiz_sum)
{
	const double a = log_integrand_amax(v, lp, n, ls, l1pwz, lkiz_sum);

	return std::lgamma(v*a) + (a-1.0) * lp - n*std::lgamma(a) - v*a*(ls + l1pwz)
	       + (a-1) * lkiz_sum;
}


/*
 * The innermost integrand of the double integral; the integrand in `a`.
 */
template<bool log_integrand>
double inner_integrand_template(const double a, const double lp,
                                const double ls, const double n,
                                const double v, const double l1p_kiz_sum,
                                const double l1p_wz, const double log_scale
                                )
{
	const double va = v * a;

	/*
	 * Shortcut for small a
	 * With SymPy, we find the following limit for a -> 0:
	 *    loggamma(v*a) - v*n*loggamma(a) --> -inf * sign(n - 1)
	 * Since for finite data sets, n>1 (the number of data points
	 * will be added), we hence have lS -> -inf and thus S -> 0
	 */
	if (a == 0)
		return 0;
	double lga = std::lgamma(a);
	if (std::isinf(lga))
		return 0;
	double lgva = std::lgamma(va);
	if (std::isinf(lgva))
		return 0;


	// Term PROD_i{  (1-k[i] * z) ^ (a-1)  }
	double lS = (a-1) * l1p_kiz_sum;

	// Term ( 1 / (s_new * (1-w*z))) ^ va
	lS -= va * (ls + l1p_wz);

	// Remaining summands:
	lS += lgva + (a-1.0) * lp - n * lga - log_scale;

	// Shortcut for debugging purposes:
	if (log_integrand)
		return lS;

	// Compute the result and test for finiteness:
	double result = exp(lS);
	if (std::isinf(result)){
		throw ScaleError("inner_integrand", lS);
	}

	if (std::isnan(result)){
		std::string msg("inner_integrand: NaN result at a =");
		msg.append(std::to_string(a));
		throw std::runtime_error(msg);
	}

	return result;
}


static double outer_integrand(double z, const double lp,
                              const double ls, const double n,
                              const double v, const std::vector<double>& ki,
                              const double w, const double log_scale)
{
	/* exp_sinh tolerance: */
	constexpr double tol = cnst_sqrt(std::numeric_limits<double>::epsilon());

	// Set the inner parameters:
	double l1p_kiz_sum = 0.0;
	for (double k : ki)
		l1p_kiz_sum += std::log1p(-k * z);
	const double l1p_wz = std::log1p(-w * z);

	/* The non-degenerate case.
	 * First compute the maximum of the a integrand at this given z:
	 */
	const double lImax = log_integrand_maximum(v, lp, n, ls, l1p_wz,
	                                           l1p_kiz_sum);

	// Integrate:
	auto integrand = [=](double a) -> double {
		return inner_integrand_template<false>(a, lp, ls, n, v,
		                                       l1p_kiz_sum, l1p_wz,
		                                       lImax);
	};
	double error, L1;
	size_t levels;
	exp_sinh<double> integrator;
	double S = integrator.integrate(integrand, tol, &error, &L1, &levels);

	if (std::isinf(S)){
		throw ScaleError("outer_integrand", log_scale);
	}

	/* Error checking: */
	if (error > 1e-5 * L1){
		/* Check if this is relevant: */
		throw PrecisionError("outer_integrand", error, L1);
	}

	return S * exp(lImax - log_scale);
}



struct az_t {
	double a;
	double z;
	double log_integrand;
};

/*
 * Find the maximum of the inner integrand across all a & z
 */
static az_t log_integrand_max(const double lp, const double ls, const double n,
                              const double v, const double w,
                              const std::vector<double>& ki)
{
	/* Start from the middle of the interval: */
	double z = 0.5;
	double l1p_kiz_sum = 0.0;

	for (uint_fast8_t i=0; i<20; ++i){
		/* Set the new parameters:: */
		l1p_kiz_sum = 0.0;
		for (double k : ki)
			l1p_kiz_sum += std::log1p(-k * z);
		double l1p_wz = std::log1p(-w * z);

		/* First and second derivatives of the above by z: */
		double k_1mkz_sum = 0.0;
		double k2_1mkz2_sum = 0.0;
		for (double k : ki){
			double x = k / (1.0 - k*z);
			k_1mkz_sum -= x;
			k2_1mkz2_sum -= x*x;
		}
		double w_1mwz = -w / (1.0 - w*z);
		double w2_1mwz2 = - w_1mwz * w_1mwz;


		/* New amax: */
		double amax = log_integrand_amax(v, lp, n, ls, l1p_wz, l1p_kiz_sum);

		/* Log of integrand:
		 * f0 =   std::lgamma(v*amax) + (amax - 1.0) * lp
		 *           - n*std::lgamma(amax) - v*amax*(ls + l1p_wz)
		 *           + (amax - 1.0) * l1p_kiz_sum
		 */

		/* Derivative of the log of the integrand by z: */
		double f1 = - v * amax * w_1mwz + (amax - 1.0) * k_1mkz_sum;

		/* Second derivative of the log of the integrand by z: */
		double f2 = - v * amax * w2_1mwz2 + (amax - 1.0) * k2_1mkz2_sum;

		/* Newton step: */
		z = std::min(std::max(z - f1 / f2, 0.0), 1.0 - 1e-8);
	}

	/* Determine amax for the final iteration: */
	l1p_kiz_sum = 0.0;
	for (double k : ki)
		l1p_kiz_sum += std::log1p(-k * z);
	double l1p_wz = std::log1p(-w * z);
	double amax = log_integrand_amax(v, lp, n, ls, l1p_wz, l1p_kiz_sum);

	/* Log of the integrand: */
	double f0 = std::lgamma(v*amax) + (amax - 1.0) * lp
	            - n*std::lgamma(amax) - v*amax*(ls + l1p_wz)
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

static double C1_computations(const double a, const double v, const double w,
                              const double h0, const double h1,
                              const unsigned char derivative=0)
{
	const double h1h0 = h1 / h0;
	if (derivative == 0)
		return (h1h0 + v*w/(w-1))*a - h1h0;
	else if (derivative == 1)
		return h1h0 + v*w/(w-1);

	return 0;
}

static double C2_computations(const double a, const double v, const double w,
                              const double h0, const double h1, const double h2,
                              const unsigned char derivative=0)
{
	const double h1h0 = h1 / h0;
	const double h2h0 = h2 / h0;
	const double nrm = 1.0 / (w*w - 2*w + 1);
	const double D2 = (v*v*w*w + 2*h1h0*v*w*(w - 1) + h1h0*h1h0*w*(w - 2)
	                   + h1h0 * h1h0) * nrm;
	const double D1 = (v*w*w - 2*h1h0*w*v*(w - 1) + 2*h2h0*(w*(w - 2) + 1)
	                   - 3*h1h0*h1h0*(w*(w-2)+1)) * nrm;
	const double D0 = 2*(h1h0*h1h0 - h2h0);
	if (derivative == 0)
		return D2*a*a + D1 * a + D0;
	else if (derivative == 1)
		return 2*D2*a + D1;
	else if (derivative == 2)
		return 2*D2;
	return 0.0;
}

template<unsigned char derivative>
static double C3_computations(const double a, const double v, const double w,
                              const double h0, const double h1, const double h2,
                              const double h3)
{
	const double h1h0 = h1 / h0;
	const double h2h0 = h2 / h0;
	const double h3h0 = h3 / h0;
	const double v2 = v*v;
	const double v3 = v2*v;
	const double w2 = w*w;
	const double w3 = w2*w;
	const double nrm = 1/(w3 - 3*w2 + 3*w - 1);
	const double D3 = (v3*w3 + 3*h1h0*v2*w2*(w-1) + 3*h1h0*h1h0*v*w*(w*(w-2)+1)
	                   + h1h0*h1h0*h1h0*(w*(w*w - 3*w + 3) - 1)) * nrm;

	if (derivative == 3)
		return 6*D3;

	const double D2 = 3*(v2*w3 + h1h0*v*w2*(v-1)*(1-w) + 2*h2h0*v*w*(w-1)*(w-1)
	                     + 3*h1h0*h1h0*v*w2*(2-w) - 3*h1h0*h1h0*v*w
	                     + 2*h1h0*h2h0*(w*(w2 - 3*w + 3) - 1)
	                     - 2*h1h0*h1h0*h1h0*(w*(w2 - 3*w + 3) -1)) * nrm;

	if (derivative == 2)
		return 6*D3*a + 2*D2;

	const double D1 = (2*v*w3 + 3*v*w*(w*(h1h0*(1-w) + 2*h2h0*(2-w)) - 2*h2h0)
	                   + 6*h3h0*(w3 - 3*w2 + 3*w - 1)
	                   + 6*h1h0*h1h0*v*w*(w2 - 2*w + 1)
	                   + 18*h1h0*h2h0*(-w3 + 3*w2 - 3*w + 1)
	                   + 11*h1h0*h1h0*h1h0*(w3 - 3*w2 + 3*w - 1)) * nrm;

	if (derivative == 1)
		return 3*D3*a*a + 2*D2*a + D1;

	const double D0 = 6*(2*h1h0*h2h0 - h3h0 - h1h0*h1h0*h1h0);

	if (derivative == 0)
		return D3*a*a*a + D2 * a*a + D1 * a + D0;

	return 0.0;
}


template<unsigned char order>
double large_z_amax(const double lp, const double ls, const double n,
                    const double v, const double h0, const double h1,
                    const double h2, const double h3, const double w,
                    const double l1pw, const double ym)
{
	/* This method computes max location of the maximum of the four integrals
	 * used in the Taylor expansion of the double integral for large z. */
	constexpr double TOL = 1e-14;
	const double lh0 = std::log(h0);
	const double lym = std::log(ym);
	double amax = (n > v) ? std::max((lp - v*ls + v*log(v) + lh0 - v * l1pw
	                                  + lym) / (n - v), 1.0)
	                      : 1.0;

	/* Recurring terms of the first and second derivatives of the a-integrands:
	 */
	auto f0_base = [&](double a) -> double {
		return v * digamma(v*a) + lp - n*digamma(a) - v*ls + lh0 - v*l1pw;
	};
	auto f1_base = [&](double a) -> double {
		return v*v*trigamma(v*a) - n*trigamma(a);
	};

	if (order == 0){
		for (int i=0; i<200; ++i){
			const double f0 = f0_base(amax) - 1/amax + lym;
			const double f1 = f1_base(amax) + 1/(amax*amax);
			const double da = std::max(-f0/f1, -0.9*amax);
			amax += da;
			if (std::abs(da) < TOL*std::abs(amax))
				break;
		}
	} else if (order == 1){
		for (int i=0; i<200; ++i){
			const double C1   = C1_computations(amax, v, w, h0, h1, 0);
			const double C1_1 = C1_computations(amax,v,w,h0,h1,1)/C1;
			const double C1_2 = C1_computations(amax,v,w,h0,h1,2)/C1
			                    - C1_1*C1_1;
			const double f0 = f0_base(amax) - 1/(amax+1) + lym + C1_1;
			const double f1 = f1_base(amax) + 1/((amax+1)*(amax+1)) + C1_2;
			const double da = std::max(-f0/f1, -0.9*amax);
			amax += da;
			if (std::abs(da) < TOL*std::abs(amax))
				break;
		}
	} else if (order == 2){
		for (int i=0; i<200; ++i){
			const double C2 = C2_computations(amax, v, w, h0, h1, h2, 0);
			const double C2_1 = C2_computations(amax, v, w, h0, h1, h2, 1)/C2;
			const double C2_2 = C2_computations(amax, v, w, h0, h1, h2, 2)/C2
			                    - C2_1*C2_1;
			const double f0 = f0_base(amax) - 1/(amax+2) + lym + C2_1;
			const double f1 = f1_base(amax) + 1/((amax+2)*(amax+2)) + C2_2;
			const double da = std::max(-f0/f1, -0.9*amax);
			amax += da;
			if (std::abs(da) < TOL*std::abs(amax))
				break;
		}
	} else if (order == 3){
		for (int i=0; i<200; ++i){
			const double C3 = C3_computations<0>(amax, v, w, h0, h1, h2, h3);
			const double C3_1 = C3_computations<1>(amax,v,w,h0,h1,h2,h3)/C3;
			const double C3_2 = C3_computations<2>(amax,v,w,h0,h1,h2,h3)/C3
			                    - C3_1*C3_1;
			const double f0 = f0_base(amax) - 1/(amax+3) + lym + C3_1;
			const double f1 = f1_base(amax) + 1/((amax+3)*(amax+3)) + C3_2;
			const double da = std::max(-f0/f1, -0.9*amax);
			amax += da;
			if (std::abs(da) < TOL*std::abs(amax))
				break;
		}
	}
	return amax;
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

struct log_double_t {
	double log_abs;
	short sign;
};


template<C_t C, bool y_integrated>
log_double_t
a_integral_large_z_log_integrand(double a, double ly, double log_scale,
                            const double lp, const double ls, const double n,
                            const double v, const double w, const double h0,
                            const double h1, const double h2, const double h3,
                            const double l1p_w, const double lh0)
{
	const double va = v * a;

	// Compute C:
	double lC = 0.0;
	double sign = 1.0;
	if (C == C_t::C0){
		/* C0 = 1 */
		lC = 0.0;
	} else if (C == C_t::C1) {
		/* C1 */
		const double C1 = C1_computations(a, v, w, h0, h1);
		if (C1 < 0){
			sign = -1.0;
			lC = std::log(-C1);
		} else {
			lC = std::log(C1);
		}
	} else if (C == C_t::C2){
		/* C2 */
		const double C2 = C2_computations(a, v, w, h0, h1, h2);
		if (C2 < 0){
			sign = -1.0;
			lC = std::log(-C2);
		} else {
			lC = std::log(C2);
		}
	} else if (C == C_t::C3){
		/* C3 */
		const double C3 = C3_computations<0>(a, v, w, h0, h1, h2, h3);
		if (C3 < 0){
			sign = -1.0;
			lC = std::log(-C3);
		} else {
			lC = std::log(C3);
		}
	}

	/* Check if we might want to return -inf: */
	if (a == 0)
		return {.log_abs=-std::numeric_limits<double>::infinity(),
		        .sign=(short)sign};
	const double lgva = std::lgamma(va);
	const double lga = std::lgamma(a);
	if (std::isinf(lgva) || std::isinf(lga))
		return {.log_abs=-std::numeric_limits<double>::infinity(),
		        .sign=(short)sign};

	// Term
	//     C * y^(a+m) / (a+m)
	// from the y power integration or
	//     C * y^(a+m-1)
	// if not integrated
	constexpr unsigned char m = C;
	double lS = lC;
	if (y_integrated){
		// Integrated:
		lS += (a + m) * ly - std::log(a+m);
	} else {
		// Not integrated:
		lS += (a + m - 1) * ly;
	}

	// Term ( s_tilde / (1-w) ) ^ va
	lS -= va * (ls + l1p_w);

	// Remaining summands:
	lS += lgva + (a-1.0) * (lp + lh0) - n * lga - log_scale;

	if (std::isinf(lS)){
		return {.log_abs=-std::numeric_limits<double>::infinity(),
		        .sign=(short)sign};
	}

	return {.log_abs=lS, .sign=(short)sign};
}

template<C_t C, bool y_integrated>
double a_integral_large_z_integrand(double a, double ly, double log_scale,
                            const double lp, const double ls, const double n,
                            const double v, const double w, const double h0,
                            const double h1, const double h2, const double h3,
                            const double l1p_w, const double lh0)
{
	auto res = a_integral_large_z_log_integrand<C,y_integrated>(a, ly,
	                        log_scale, lp, ls, n, v, w, h0, h1, h2, h3, l1p_w,
	                        lh0);

	// Compute the result and test for finity:
	double result = std::exp(res.log_abs);
	if (std::isinf(result)){
		std::cout << "log_abs = " << res.log_abs << "\n" << std::flush;
		if (C == C_t::C0)
			throw ScaleError("a_integral_large_z_integrand_0", res.log_abs);
		else if (C == C_t::C1)
			throw ScaleError("a_integral_large_z_integrand_1", res.log_abs);
		else if (C == C_t::C2)
			throw ScaleError("a_integral_large_z_integrand_2", res.log_abs);
		else if (C == C_t::C3)
			throw ScaleError("a_integral_large_z_integrand_3", res.log_abs);
	}

	return result;
}


static double y_taylor_transition_root_backend(double y, const double lp,
                            const double ls, const double n, const double v,
                            const double h0, const double h1, const double h2,
                            const double h3, const double lh0, const double w,
                            const double l1p_w, const double log_scale0)
{
	/* exp_sinh tolerance:
	 * Since we only care about a rough logarithmic comparison,
	 * we do not need very high precision in these integrals.
	 **/
	constexpr double tol = 1e-3;
	constexpr double epsilon = 1e-14;

	/* Get the scale: */
	const double amax = large_z_amax<0>(lp, ls, n, v, h0, h1, h2, h3,
	                                    w, l1p_w, y);
	const double ly = std::log(y);
	const double log_scale = log_scale0
	    + a_integral_large_z_log_integrand<C_t::C0,true>(amax, ly, log_scale0,
	                                                lp,  ls, n, v, w, h0, h1,
	                                                h2, h3, l1p_w,
	                                                lh0).log_abs;

	/* Compute the 'a' integrals for the constant and the cubic term: */
	double error, L1;
	size_t levels;
	exp_sinh<double> integrator;

	auto integrand0 = [&](double a) -> double
	{
		return a_integral_large_z_integrand<C_t::C0,true>(a, ly, log_scale, lp,
		                                             ls, n, v, w, h0, h1, h2,
		                                             h3, l1p_w, lh0);
	};

	double S0 = integrator.integrate(integrand0, tol, &error, &L1, &levels);

	if (std::isinf(S0)){
		throw ScaleError("y_taylor_transition_root_backend_S0", log_scale);
	}
	/* Error checking: */
	if (error > 1e-2 * L1){
		throw PrecisionError("y_taylor_transition_root_backend_S0", error, L1);
	}


	auto integrand1 = [&](double a) -> double
	{
		return a_integral_large_z_integrand<C_t::C3,true>(a, ly, log_scale, lp,
		                                             ls, n, v, w, h0, h1, h2,
		                                             h3, l1p_w, lh0);
	};

	double S1 = integrator.integrate(integrand1, tol, &error, &L1, &levels);

	if (std::isinf(S1)){
		throw ScaleError("y_taylor_transition_root_backend_S1", log_scale);
	}
	/* Error checking: */
	if (error > 1e-2 * std::max(L1,std::fabs(S0))){
		throw PrecisionError("y_taylor_transition_root_backend_S1", error, L1);
	}

	/* Extract the result: */
	const double result = std::log(std::fabs(S1)) - std::log(epsilon)
	                      - std::log(std::fabs(S0));

	/* Make sure that result is finite: */
	if (std::isinf(result)){
		throw ScaleError("y_taylor_transition_root_backend", 300.);
	}

	return result;
}




static double y_taylor_transition(const double h0, const double h1,
                 const double h2, const double h3, const double nu_new,
                 const double lp_tilde, const double n_new,
                 const double log_scale, const double ls_tilde,
                 const double l1p_w, const double w,
                 const double ymin=1e-32)
{
	/* Find a value above the threshold: */
	double yr = 1e-9;
	double lh0 = std::log(h0);
	double val = y_taylor_transition_root_backend(yr, lp_tilde, ls_tilde,
	                                              n_new, nu_new, h0, h1, h2,
	                                              h3, lh0, w, l1p_w,
	                                              log_scale);
	while (val < 0 || std::isnan(val)){
		yr = std::min(2*yr, 1.0);
		if (yr == 1.0)
			break;
		val = y_taylor_transition_root_backend(yr, lp_tilde, ls_tilde, n_new,
		                                       nu_new, h0, h1, h2, h3, lh0, w,
		                                       l1p_w, log_scale);
	}

	/* Root finding: */
	auto rootfun = [&](double y) -> double {
		return y_taylor_transition_root_backend(y, lp_tilde, ls_tilde, n_new,
		                                        nu_new, h0, h1, h2, h3, lh0,
		                                        w, l1p_w, log_scale);
	};
	constexpr std::uintmax_t MAX_ITER = 100;
	std::uintmax_t max_iter = MAX_ITER;
	eps_tolerance<double> tol(2);

	std::pair<double,double> bracket
	   = toms748_solve(rootfun, ymin, yr, tol, max_iter);

	if (max_iter >= MAX_ITER)
		throw std::runtime_error("Could not determine root.");

	return 0.5 * (bracket.first + bracket.second);
}



/*
 * Compute the actual integral:
 */
template<bool y_integrated>
double a_integral_large_z(const double ym, const double h0, const double h1,
                 const double h2, const double h3, const double nu_new,
                 const double lp_tilde, const double n_new,
                 const double log_scale, const double ls_tilde,
                 const double l1p_w, const double w)
{
	/* exp_sinh tolerance: */
	constexpr double tol = cnst_sqrt(std::numeric_limits<double>::epsilon());

	/* Set the integrand's non-varying parameters: */
	const double ly = std::log(ym);
	const double lh0 = std::log(h0);

	/* Integration setup for 'a' integrals:: */
	double error, L1;
	size_t levels;
	exp_sinh<double> integrator;

	auto error_check = [&](double S, C_t C, double S_ref){
		if (std::isinf(S)){
			if (C == C_t::C0)
				throw ScaleError("a_integral_large_z_S0", log_scale);
			else if (C == C_t::C1)
				throw ScaleError("a_integral_large_z_S1", log_scale);
			else if (C == C_t::C2)
				throw ScaleError("a_integral_large_z_S2", log_scale);
			else if (C == C_t::C3)
				throw ScaleError("a_integral_large_z_S3", log_scale);
		}
		/* Error checking: */
		if (error > 1e-5 * std::max(L1,S_ref)){
			if (C == C_t::C0)
				throw PrecisionError("a_integral_large_z_S0", error, L1);
			if (C == C_t::C1)
				throw PrecisionError("a_integral_large_z_S1", error, L1);
			if (C == C_t::C2)
				throw PrecisionError("a_integral_large_z_S2", error, L1);
			if (C == C_t::C3)
				throw PrecisionError("a_integral_large_z_S3", error, L1);
		}
	};

	/*
	 *  Constant C0.
	 */
	auto integrand0 = [&](double a) -> double
	{
		return a_integral_large_z_integrand<C_t::C0, y_integrated>(a, ly,
		                           log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		                           w, h0, h1, h2, h3, l1p_w, lh0);
	};
	double S_ref = 0.0;
	double S0 = integrator.integrate(integrand0, tol, &error, &L1, &levels);
	error_check(S0, C_t::C0, 0.0);
	S_ref = std::fabs(S0);

	/*
	 *  Constant C1.
	 */
	auto integrand1 = [&](double a) -> double
	{
		return a_integral_large_z_integrand<C_t::C1,y_integrated>(a, ly,
		                           log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		                           w, h0, h1, h2, h3, l1p_w, lh0);
	};
	double S1 = integrator.integrate(integrand1, tol, &error, &L1, &levels);
	error_check(S1, C_t::C1, S_ref);
	S_ref = std::max(S_ref, std::fabs(S1));

	/*
	 *  Constant C2.
	 */
	auto integrand2 = [&](double a) -> double
	{
		return a_integral_large_z_integrand<C_t::C2,y_integrated>(a, ly,
		                           log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		                           w, h0, h1, h2, h3, l1p_w, lh0);
	};

	double S2 = integrator.integrate(integrand2, tol, &error, &L1, &levels);
	error_check(S2, C_t::C2, S_ref);
	S_ref = std::max(S_ref, std::fabs(S2));

	/*
	 *  Constant C3.
	 */
	auto integrand3 = [&](double a) -> double
	{
		return a_integral_large_z_integrand<C_t::C3,y_integrated>(a, ly,
		                           log_scale, lp_tilde, ls_tilde, n_new, nu_new,
		                           w, h0, h1, h2, h3, l1p_w, lh0);
	};

	double S3 = integrator.integrate(integrand3, tol, &error, &L1, &levels);
	error_check(S3, C_t::C3, S_ref);


	return S0 + S1 + S2 + S3;
}




/*
 * Compute the posterior:
 */

using pdtoolbox::heatflow::posterior_t;


static void init_locals(const double* qi, const double* ci, size_t N,
                        double nu, double n, double s, double p,
                        // Destination parameters:
                        double& lp_tilde, double& ls_tilde, double& nu_new,
                        double& n_new, double& Qmax, double& h0, double& h1,
                        double& h2, double& h3, double& w, double& l1p_w,
                        double& ztrans, double& log_scale, double& norm,
                        std::vector<double>& ki, double& full_taylor_integral,
                        size_t& imax)
{
	// Initialize parameters:
	nu_new = nu + N;
	n_new = n + N;

	// Step 0: Determine Qmax, A, B, and ki:
	imax = 0;
	Qmax = std::numeric_limits<double>::infinity();
	for (size_t i=0; i<N; ++i){
		if (qi[i] <= 0)
			throw std::runtime_error("At least one qi is zero or negative and "
			                         "has hence left the model definition "
			                         "space.");
		double Q = qi[i] / ci[i];
		if (Q > 0 && Q < Qmax){
			Qmax = Q;
			imax = i;
		}
	}
	if (std::isinf(Qmax))
		throw std::runtime_error("Found Qmax == inf. The model is not "
		                         "well-defined. This might happen if all "
		                         "ci <= 0, i.e. heat flow anomaly has a "
		                         "negative or no impact on all data points.");

	double A = s;
	for (size_t i=0; i<N; ++i)
		A += qi[i];
	ls_tilde = std::log(A);
	double B = 0.0;
	for (size_t i=0; i<N; ++i)
		B += ci[i];
	for (size_t i=0; i<N; ++i)
		ki[i] = ci[i] * Qmax / qi[i];
	double lq_sum = 0.0;
	for (size_t i=0; i<N; ++i)
		lq_sum += std::log(qi[i]);

	lp_tilde = std::log(p) + lq_sum;
	w = B * Qmax / A;

	// Integration config:

	/* Get an estimate of the scaling: */

	/* The new version: */
	az_t azmax = log_integrand_max(lp_tilde, ls_tilde, n_new, nu_new, w, ki);
	log_scale = azmax.log_integrand;

	/* Compute the coefficients for the large-z (small-y) Taylor expansion: */
	h0=1.0;
	h1=0;
	h2=0;
	h3=0;
	for (size_t i=0; i<N; ++i){
		if (i == imax)
			continue;
		const double d0 = 1.0 - ki[i];
		h3 = d0 * h3 + ki[i] * h2;
		h2 = d0 * h2 + ki[i] * h1;
		h1 = d0 * h1 + ki[i] * h0;
		h0 *= d0;
	}

	/* Step 1: Compute the normalization constant.
	 *         This requires the full integral and might require
	 *         a readjustment of the log_scale parameter. */
	l1p_w = std::log1p(-w);
	bool norm_success = false;
	full_taylor_integral = std::nan("");
	norm = std::nan("");
	ztrans = std::nan("");
	double ymax, S;

	auto integrand = [&](double z) -> double
	{
		return outer_integrand(z, lp_tilde, ls_tilde, n_new, nu_new, ki, w,
		                       log_scale);
	};

	while (!norm_success && !std::isinf(log_scale)){
		// Compute the transition z where we switch to an analytic integral of
		// the Taylor expansion:
		try {
			ymax = y_taylor_transition(h0, h1, h2, h3, nu_new, lp_tilde, n_new,
			                           log_scale, ls_tilde, l1p_w, w, 1e-32);
		} catch (ScaleError& s) {
			if (s.log_scale() < 0){
				std::string msg("Failed to determine the normalization "
				                "constant: Encountered negative scale error "
				                "in ");
				msg.append(s.what());
				msg.append(": ");
				msg.append(std::to_string(s.log_scale()));
				throw std::runtime_error(msg);
			}
			log_scale += s.log_scale();
			continue;
		} catch (PrecisionError& s){
			std::string msg("Failed to determine the normalization "
			                "constant: Encountered precision error "
			                "in ");
			msg.append(s.what());
			msg.append(": error=");
			msg.append(std::to_string(s.error()));
			msg.append(", L1=");
			msg.append(std::to_string(s.L1()));
			throw std::runtime_error(msg);
		}
		ztrans = 1.0 - ymax;

		try {
			// 1.1: Double numerical integration in z range [0,1-ymax]:
			double error;
			tanh_sinh<double> integrator;
			S = integrator.integrate(integrand, 0, ztrans, 1e-12, &error);

		} catch (ScaleError& s) {
			if (s.log_scale() < 0)
				throw std::runtime_error("Failed to determine the "
				                         "normalization constant: Encountered "
				                         "negative scale error.");
			log_scale += s.log_scale();
			continue;
		} catch (PrecisionError& s) {
			s.append_message("Failed to determine the normalization "
			                 "constant (moderate z).");
			throw s;
		} catch (std::runtime_error& err){
			std::string msg("Failed to determine the normalization constant "
			                "for the integrand: '");
			msg.append(err.what());
			msg.append("'.\nMore information: ztrans=");
			msg.append(std::to_string(ztrans));
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
			                   n_new, log_scale, ls_tilde, l1p_w, w);
			S += full_taylor_integral;
		} catch (ScaleError& s) {
			if (s.log_scale() < 0)
				throw std::runtime_error("Failed to determine the "
				                         "normalization constant: Encountered "
				                         "negative scale error.");
			log_scale += s.log_scale();
			continue;
		} catch (PrecisionError& s) {
			s.append_message("Failed to determine the normalization constant "
			                 "(large z).");
			throw s;
		} catch (std::runtime_error& err){
			std::string msg("Failed to determine the normalization constant "
			                "for the integrand: '");
			msg.append(err.what());
			msg.append("'.\nMore information: ztrans=");
			msg.append(std::to_string(ztrans));
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
               double nu, double dest_tol)
{
	/* Computes the posterior for a given parameter combination using
	 * two-dimensional adaptive quadrature.
	 */

	/* Step 1: Use the common parameter initialization routine which
	 *         mainly determines the normalization constant: */
	double lp_tilde=0, ls_tilde=0, nu_new=0, n_new=0, Qmax=0, h0=0, h1=0,
	       h2=0, h3=0, w=0, l1p_w=0, ztrans=0, log_scale=0, norm=0,
	       full_taylor_integral=0;
	size_t imax;
	std::vector<double> ki(N);
	init_locals(qi, ci, N, nu, n, s, p, lp_tilde,
	            ls_tilde, nu_new, n_new, Qmax, h0, h1, h2, h3, w, l1p_w,
	            ztrans, log_scale, norm, ki, full_taylor_integral, imax);

	std::vector<size_t> order(0);
	bool is_cumulative = type & (posterior_t::CUMULATIVE | posterior_t::TAIL);
	double z_last = 0.0, S_cumul = 0.0;
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

	auto integrand = [&](double z) -> double {
		return outer_integrand(z, lp_tilde, ls_tilde, n_new, nu_new, ki, w,
		                       log_scale);
	};

	/* A condition for deciding the integrator: */
	bool use_thsh = (integrand(ztrans)*ztrans > 2*norm);

	tanh_sinh<double> thsh_int;
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
		const double zi = x[j] / Qmax;
		if (type == posterior_t::DENSITY){
			/* Return the density in P_H (aka \dot{Q}) of the posterior.
			 * To do so, return the outer integrand at z corresponding to the
			 * requested x[i].
			 * Then norm the outer_integrand to a PDF in z. Revert the change of
			 * variables x->z by dividing the PDF by Qmax.
			 */
			if (zi <= ztrans)
				res[j] = integrand(zi) / (Qmax * norm);
			else
				// Use the Taylor expansion for the limit z->1, y=1-z -> 0:
				res[j] = a_integral_large_z<false>(1.0-zi, h0, h1, h2, h3,
				                       nu_new, lp_tilde, n_new, log_scale,
				                       ls_tilde, l1p_w, w)
				         / (Qmax * norm);
		} else if (type == posterior_t::CUMULATIVE){
			double S;
			if (zi <= ztrans){
				double err;
				constexpr double eps_tol
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
					double norm_num = S + S_cumul + full_taylor_integral;
					if (std::fabs(norm_num - norm) > dest_tol * norm){
						throw PrecisionError("CDF normalization failed to "
						                     "achieve the desired precision.",
						                     std::fabs(norm_num - norm), norm);
					}
					/* Renormalize: */
					const double rescale = 1.0 / norm_num;
					for (size_t k=0; k<Nx; ++k)
						res[k] *= rescale;
				}
			} else {
				/* Sanity check: */
				if (z_last < ztrans){
					S = GK::integrate(integrand, z_last, ztrans);
					double norm_num = S + S_cumul + full_taylor_integral;
					if (std::fabs(norm_num - norm) > dest_tol * norm){
						throw PrecisionError("CDF normalization failed to "
						                     "achieve the desired precision.",
						                     std::fabs(norm_num - norm), norm);
					}
				}
				/* Now integrate from the back: */
				S = norm
				    - a_integral_large_z<true>(1.0-zi, h0, h1, h2, h3, nu_new,
				                       lp_tilde, n_new, log_scale, ls_tilde,
				                       l1p_w, w);
				S_cumul += S;
				z_last = zi;
			}
			res[j] = std::max(std::min(S_cumul / norm, 1.0), 0.0);
		} else if (type == posterior_t::TAIL){
			double S;
			if (zi <= ztrans){
				// First the part from zi to ztrans:
				if (z_last >= ztrans){
					z_last = ztrans;
					S_cumul = full_taylor_integral;
				}
				if (use_thsh)
					S = thsh_int.integrate(integrand, zi, z_last);
				else {
					double err;
					constexpr double eps_tol
					   = boost::math::tools::root_epsilon<double>();
					if (zi == z_last){
						S = 0.0;
					} else {
						S = GK::integrate(integrand, zi, z_last, 1,
						                  eps_tol, &err);
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
				S_cumul = a_integral_large_z<true>(1.0-zi, h0, h1, h2, h3,
				                       nu_new, lp_tilde, n_new, log_scale,
				                       ls_tilde, l1p_w, w);
				z_last = zi;
			}
			res[j] = std::max(std::min(S_cumul / norm, 1.0), 0.0);

			/* Sanity check of the normalization: */
			if (i+1 == Nx && z_last < ztrans){
				if (z_last > 0)
					if (use_thsh)
						S = thsh_int.integrate(integrand, 0, z_last);
					else
						S = GK::integrate(integrand, 0, z_last);
				else
					S = 0.0;
				double norm_num = S + S_cumul;
				if (std::fabs(norm_num - norm) > dest_tol * norm){
					throw PrecisionError("Tail distribution normalization "
					                     "failed to achieve the desired "
					                     "precision.",
					                     std::fabs(norm_num - norm), norm);
				}
				/* Renormalize: */
				const double rescale = norm / norm_num;
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
				res[j] = std::log(integrand(zi)) + log_scale;
			} else {
				res[j] = std::log(a_integral_large_z<false>(1.0-zi, h0, h1, h2,
				                       h3, nu_new, lp_tilde, n_new, log_scale,
				                       ls_tilde, l1p_w, w))
				         + log_scale;
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
                   double nu, double dest_tol)
{
	posterior<DENSITY>(x, res, Nx, qi, ci, N, p, s, n, nu, dest_tol);
}


void posterior_pdf_batch(const double* x, size_t Nx, double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double dest_tol)
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
		try {
			posterior_pdf(x, res + Nx*i, Nx, qi[i], ci[i], N[i], p, s, n, nu,
			              dest_tol);
		} catch (const std::exception& e){
			error_flag = true;
			err_msg = std::string(e.what());
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
                   double nu, double dest_tol)
{
	posterior<CUMULATIVE>(x, res, Nx, qi, ci, N, p, s, n, nu, dest_tol);
}


void posterior_cdf_batch(const double* x, size_t Nx, double* res,
                         const std::vector<const double*>& qi,
                         const std::vector<const double*>& ci,
                         const std::vector<size_t>& N,
                         double p, double s, double n, double nu,
                         double dest_tol)
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
		try {
			posterior_cdf(x, res + Nx*i, Nx, qi[i], ci[i], N[i], p, s, n, nu,
			              dest_tol);
		} catch (const std::exception& e){
			error_flag = true;
			err_msg = std::string(e.what());
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
                    double nu, double dest_tol)
{
	posterior<TAIL>(x, res, Nx, qi, ci, N, p, s, n, nu, dest_tol);
}


void posterior_tail_batch(const double* x, size_t Nx, double* res,
                          const std::vector<const double*>& qi,
                          const std::vector<const double*>& ci,
                          const std::vector<size_t>& N,
                          double p, double s, double n, double nu,
                          double dest_tol)
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
		try {
			posterior_tail(x, res + Nx*i, Nx, qi[i], ci[i], N[i], p, s, n, nu,
			               dest_tol);
		} catch (const std::exception& e){
			error_flag = true;
			err_msg = std::string(e.what());
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
                            double dest_tol)
{
	posterior<UNNORMED_LOG>(x, res, Nx, qi, ci, N, p, s, n, nu, dest_tol);
}


/*
 * A version of the above catching errors and returning NaNs:
 */
void posterior_silent(const double* x, double* res, size_t Nx, const double* qi,
               const double* ci, size_t N, double p, double s, double n,
               double nu, double dest_tol, posterior_t type)
{
	try {
		if (type == DENSITY)
			posterior<DENSITY>(x, res, Nx, qi, ci, N, p, s, n, nu, dest_tol);
		else if (type == CUMULATIVE)
			posterior<CUMULATIVE>(x, res, Nx, qi, ci, N, p, s, n, nu,
			                      dest_tol);
		else if (type == TAIL)
			posterior<TAIL>(x, res, Nx, qi, ci, N, p, s, n, nu, dest_tol);
		else if (type == UNNORMED_LOG)
			posterior<UNNORMED_LOG>(x, res, Nx, qi, ci, N, p, s, n, nu,
			                        dest_tol);
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
                    const double nu, const double dest_tol)
{
	/* Estimates the tail quantiles using the QuantileInverter class. */

	/* Step 1: Use the common parameter initialization routine which
	 *         mainly determines the normalization constant: */
	double lp_tilde=0, ls_tilde=0, nu_new=0, n_new=0, Qmax=0, h0=0, h1=0,
	       h2=0, h3=0, w=0, l1p_w=0, ztrans=0, log_scale=0, norm=0,
	       full_taylor_integral=0;
	size_t imax;
	std::vector<double> ki(N);
	init_locals(qi, ci, N, nu, n, s, p, lp_tilde,
	            ls_tilde, nu_new, n_new, Qmax, h0, h1, h2, h3, w, l1p_w,
	            ztrans, log_scale, norm, ki, full_taylor_integral, imax);


	auto integrand = [&](double z) -> double {
		double result = 0.0;
		if (z <= ztrans)
			result = outer_integrand(z, lp_tilde, ls_tilde, n_new, nu_new,
			                         ki, w, log_scale) / norm;
		else if (z < 1)
			result = a_integral_large_z<false>(1.0-z, h0, h1, h2, h3,
			                                   nu_new, lp_tilde, n_new,
			                                   log_scale, ls_tilde,
			                                   l1p_w, w) / norm;

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
                     double dest_tol)
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
		double full_taylor_integral;
		std::vector<double> ki;
	};
	std::vector<locals_t> locals(qi.size());
	#pragma omp parallel for
	for (size_t i=0; i<qi.size(); ++i){
		size_t imax;
		try {
			locals[i].ki.resize(N[i]);
			init_locals(qi[i], ci[i], N[i], nu, n, s, p, locals[i].lp_tilde,
			            locals[i].ls_tilde, locals[i].nu_new, locals[i].n_new,
			            locals[i].Qmax, locals[i].h0, locals[i].h1,
			            locals[i].h2, locals[i].h3, locals[i].w,
			            locals[i].l1p_w, locals[i].ztrans, locals[i].log_scale,
			            locals[i].norm, locals[i].ki,
			            locals[i].full_taylor_integral, imax);
		} catch (const std::exception& e){
			error_flag = true;
			err_msg = std::string(e.what());
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
				                          locals[i].w, locals[i].log_scale)
				          / locals[i].norm;
			else if (z < 1)
				result += a_integral_large_z<false>(1.0-z, locals[i].h0,
				                           locals[i].h1, locals[i].h2,
				                           locals[i].h3, locals[i].nu_new,
				                           locals[i].lp_tilde, locals[i].n_new,
				                           locals[i].log_scale,
				                           locals[i].ls_tilde, locals[i].l1p_w,
				                           locals[i].w) / locals[i].norm;
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
                           const double dest_tol,
                           short print)
{
	try {
		tail_quantiles(quantiles, res, Nquant, qi, ci, N, p, s, n, nu,
		               dest_tol);
	}  catch (ScaleError& e){
		if (print){
			std::cerr << "ScaleError (" << e.what() << "\n"
			             "log_scale: " << e.log_scale() << "\n";
		}
		return 1;
	} catch (PrecisionError& pe) {
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
