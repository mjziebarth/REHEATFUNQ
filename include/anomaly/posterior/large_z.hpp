/*
 * Code for computing the gamma conjugate posterior modified for heat
 * flow anomaly as described by Ziebarth & von Specht [1].
 * This code handles the transition zone for large z.
 * This code is an alternative implementation of the code found in
 * `ziebarth2022a.cpp`.
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


#ifndef REHEATFUNQ_ANOMALY_POSTERIOR_LARGE_Z_HPP
#define REHEATFUNQ_ANOMALY_POSTERIOR_LARGE_Z_HPP

/*
 * General includes:
 */
#include <cmath>
#include <limits>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include <utility>
#include <cstdint>
#include <string>


/*
 * REHEATFUNQ includes:
 */
#include <anomaly/posterior/args.hpp>
#include <anomaly/posterior/locals.hpp>
#include <anomaly/posterior/exceptions.hpp>
#include <numerics/functions.hpp>

namespace reheatfunq {
namespace anomaly {
namespace posterior {

namespace rm = reheatfunq::math;
namespace bmq = boost::math::quadrature;

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

	C1_t(typename arg<const real>::type a, const Locals<real>& L)
	{
		auto h1h0 = L.h[1] / L.h[0];
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

	C2_t(typename arg<const real>::type a, const Locals<real>& L)
	{
		auto h1h0 = L.h[1] / L.h[0];
		auto h2h0 = L.h[2] / L.h[0];
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

	C3_t(typename arg<const real>::type a, const Locals<real>& L)
	{
		auto h1h0 = L.h[1] / L.h[0];
		auto h2h0 = L.h[2] / L.h[0];
		auto h3h0 = L.h[3] / L.h[0];
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
a_integral_large_z_log_integrand(typename arg<const real>::type a,
                                 typename arg<const real>::type ly,
                                 typename arg<const real>::type log_integrand_max,
                                 const Locals<real>& L)
{
	const real va = L.v * a;

	/* A function to compute sign and logarithm of absolute of a
	 * real number:
	 */
	auto signed_log = [](typename arg<const real>::type Cx, int8_t& sign, real& lC){
		if (Cx < 0){
			sign = -1;
			lC = rm::log(-Cx);
		} else {
			lC = rm::log(Cx);
		}
	};

	// Compute C:
	real lC = 0.0;
	int8_t sign = 1;
	switch (C){
		case C0:
			lC = 0.0;
			break;
		case C1:
			signed_log(C1_t(a, L).deriv0, sign, lC);
			break;
		case C2:
			signed_log(C2_t(a, L).deriv0, sign, lC);
			break;
		case C3:
			signed_log(C3_t(a, L).deriv0, sign, lC);
			break;
	}

	/* Check if we might want to return -inf: */
	if (a == 0)
		return {.log_abs=-std::numeric_limits<real>::infinity(),
		        .sign=sign};

	// Term
	//     C * y^(a+m) / (a+m)
	// from the y power integration or
	//     C * y^(a+m-1)
	// if not integrated
	constexpr short m = C;
	if (y_integrated){
		// Integrated:
		lC += m * ly - rm::log(a+m);
	} else {
		// Not integrated:
		lC += (m - 1) * ly;
	}

	// Term ( s_tilde * (1-w) ) ^ (-va)
	lC -= va * (L.ls + L.l1p_w);

	// Remaining summands:
	lC += loggamma_v_a__minus__n_loggamma_a__plus__C_a<real>(
	          a, L.n, L.v,
	          L.lp + L.lh0 - L.v* (L.ls + L.l1p_w) + ly,
	          L.lv);
	lC -= L.lp + L.lh0 + log_integrand_max;

	if (rm::isinf(lC)){
		return {.log_abs=-std::numeric_limits<double>::infinity(),
		        .sign=sign};
	}

	return {.log_abs=lC, .sign=sign};
}

template<C_t C, bool y_integrated, typename real>
real a_integral_large_z_integrand(typename arg<const real>::type a,
                                  typename arg<const real>::type ly,
                                  typename arg<const real>::type log_integrand_max,
                                  const Locals<real>& L)
{
	log_double_t<real> res
	    = a_integral_large_z_log_integrand<C,y_integrated, real>
	           (a, ly, log_integrand_max, L);
	if (rm::isnan(res.log_abs)){
		std::string msg("NaN result in a_integral_large_z_integrand_");
		msg += std::to_string((int)C);
		msg += ".\n   at a=";
		msg += std::to_string(static_cast<long double>(a));
		msg += ".\n   at ly = ";
		msg += std::to_string(static_cast<long double>(ly));
		msg += ".\n   with log_integrand_max =  ";
		msg += std::to_string(static_cast<long double>(log_integrand_max));
		msg += ".\n   and res.log_abs =         ";
		msg += std::to_string(static_cast<long double>(res.log_abs));
		msg += ".";
		throw std::runtime_error(msg);
	}

	// Compute the result and test for finity:
	real result = rm::exp(res.log_abs);
	if (rm::isinf(result)){
		std::string msg("a_integral_large_z_integrand_");
		msg += std::to_string((int)C);
		msg += " inf result.\n   at ly = ";
		msg += std::to_string(static_cast<long double>(ly));
		msg += ".\n   at a =  ";
		msg += std::to_string(static_cast<long double>(a));
		msg += ".\n   with log_integrand_max =  ";
		msg += std::to_string(static_cast<long double>(log_integrand_max));
		msg += ".\n   and res.log_abs =         ";
		msg += std::to_string(static_cast<long double>(res.log_abs));
		msg += ".";
		throw ScaleError<real>(msg, res.log_abs);
	}

	return result;
}


template<typename real, unsigned char order>
real large_z_amax(typename arg<const real>::type ym, const Locals<real>& L)
{
	static_assert(order == 0);

	const real ly = rm::log(ym);

	auto cost = [ly,&L](typename arg<const real>::type x) -> real
	{
		real a = L.amin / (1.0 - x);
		if (rm::isinf(a))
			return std::numeric_limits<real>::infinity();
		return -a_integral_large_z_log_integrand<C_t::C0, false, real>(a, ly, 0.0, L).log_abs;
	};

	std::uintmax_t max_iter = 200;
	real xmax = boost::math::tools::brent_find_minima(
	                cost,
	                static_cast<real>(0.0),
	                static_cast<real>(1.0),
	                1000000,
	                max_iter).first;

	return L.amin / (1.0 - xmax);
}


template<typename real>
real y_taylor_transition_root_backend(typename arg<const real>::type y,
                                      const Locals<real>& L,
                                      typename arg<const real>::type outer_log_scale)
{
	/* Backup Gauss-Kronrod quadrature: */
	typedef bmq::gauss_kronrod<real,31> GK;

	constexpr double epsilon = 1e-14;

	const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();




	/* Get the scale: */
	const real amax = large_z_amax<real,0>(y, L);
	const real ly = rm::log(y);

	const real log_scale
	    = a_integral_large_z_log_integrand<C_t::C0,true>(amax, ly,
	                                                     0.0,
	                                                     L).log_abs;


	const real amax_3 = large_z_amax<real,0>(y, L);
	const real log_scale_3
	    = a_integral_large_z_log_integrand<C_t::C3,true>(amax_3, ly,
	                                                     0.0,
	                                                     L).log_abs;

	/* Crude approximation of the integral, assuming similar shapes
	 * of the integrands for C0 and C3: */
	return log_scale_3 + rm::log(amax_3) - log_scale - rm::log(amax)
	       - rm::log(std::numeric_limits<real>::epsilon());



	/* Compute the 'a' integrals for the constant and the cubic term: */
	real error, L1;
	size_t levels;
	bmq::exp_sinh<real> es_integrator;
	bmq::tanh_sinh<real> ts_integrator;

	auto integrand00 = [&](typename arg<const real>::type a,
	                       typename arg<const real>::type distance_to_border)
	                       -> real
	{
		return a_integral_large_z_integrand<C_t::C0,true,real>(
		                a, ly, log_scale, L
		);
	};

	auto integrand0 = [&](typename arg<const real>::type a) -> real
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

	if (rm::isinf(S0)){
		throw ScaleError<real>("y_taylor_transition_root_backend_S0",
		                       log_scale);
	}
	/* Error checking: */
	if (error > 1e-2 * L1){
		throw PrecisionError<real>("y_taylor_transition_root_backend_S0",
		                           error, L1);
	}

	auto integrand10 = [&](typename arg<const real>::type a,
	                       typename arg<const real>::type distance_to_border)
	                       -> real
	{
		return a_integral_large_z_integrand<C_t::C3,true,real>(
		                a, ly, log_scale, L
		);
	};

	auto integrand1 = [&](typename arg<const real>::type a) -> real
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

	if (rm::isinf(S1)){
		throw ScaleError<real>("y_taylor_transition_root_backend_S1",
		                       log_scale);
	}
	/* Error checking: */
	if (error > 1e-2 * std::max<real>(L1, rm::abs(S0))){
		throw PrecisionError<real>("y_taylor_transition_root_backend_S1",
		                           error, L1);
	}

	/* Extract the result: */
	const real result = rm::log(rm::abs(S1)) - rm::log(epsilon)
	                    - rm::log(rm::abs(S0));

	/* Make sure that result is finite: */
	if (rm::isinf(result)){
		throw ScaleError<real>("y_taylor_transition_root_backend", 300.);
	}

	return result;
}




template<typename real>
real y_taylor_transition(const Locals<real>& L,
                         typename arg<const real>::type outer_log_scale,
                         const real ymin = 1e-32)
{
	/* Find a value above the threshold: */
	real yr = 1e-20;
	real val = y_taylor_transition_root_backend<real>(yr, L, outer_log_scale);
	while (val < 0 || rm::isnan(val)){
		yr = std::min<real>(2*yr, 1.0);
		if (yr == 1.0)
			break;
		val = y_taylor_transition_root_backend<real>(yr, L, outer_log_scale);
	}

	/* Root finding: */
	auto rootfun = [&](typename arg<const real>::type y) -> real {
		return y_taylor_transition_root_backend<real>(y, L, outer_log_scale);
	};
	constexpr std::uintmax_t MAX_ITER = 100;
	std::uintmax_t max_iter = MAX_ITER;
	boost::math::tools::eps_tolerance<real> tol(2);

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
real a_integral_large_z(typename arg<const real>::type ym,
                        typename arg<const real>::type S_cmp,
                        typename arg<const real>::type log_scale,
                        const Locals<real>& L)
{
	/*
	 * When y == 0, the heat flow dat point with the largest k_i (== 1)
	 * is reduced exactly to zero. Then, this heat flow data point leaves
	 * the support. By definition, we assign probability 0 for this case.
	 *
	 * This circumvents all kinds of limit ambiguities and numerical problems
	 * involved with y == 0 (and a==1, for instance).
	 */
	if (ym == 0)
		return 0.0;

	const real TOL_TANH_SINH = boost::math::tools::root_epsilon<real>();

	/* Set the integrand's non-varying parameters: */
	const real ly = rm::log(ym);

	/* Integration setup for 'a' integrals:: */
	real error, L1, S;
	size_t levels;
	bmq::exp_sinh<real> integrator;

	auto integrand = [=,&L](real a) -> real
	{
		real S0
		   = a_integral_large_z_integrand<C_t::C0,y_integrated,real>
		         (a, ly, log_scale, L);
		real S1
		   = a_integral_large_z_integrand<C_t::C1,y_integrated,real>
		         (a, ly, log_scale, L);
		real S2
		   = a_integral_large_z_integrand<C_t::C2,y_integrated,real>
		         (a, ly, log_scale, L);
		real S3
		   = a_integral_large_z_integrand<C_t::C3,y_integrated,real>
		         (a, ly, log_scale, L);
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
	if (rm::isinf(S) || rm::isnan(S))
		throw ScaleError<real>("a_integral_large_z", 0.0);
	if (error > std::max<real>(TOL_TANH_SINH * L1, 1e-14 * S_cmp))
		throw PrecisionError<real>("a_integral_large_z_S3", error, L1);

	return S;
}

} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq

#endif
