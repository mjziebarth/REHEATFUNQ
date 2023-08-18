/*
 * Conjugate prior of the gamma distribution due to Miller (1980).
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2021 Deutsches GeoForschungsZentrum GFZ,
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
 * Miller (1980):
 */

#include <constexpr.hpp>
#include <gamma_conjugate_prior.hpp>
#include <funccache.hpp>
#include <cmath>
#include <algorithm>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>

using pdtoolbox::GammaConjugatePriorBase,
      pdtoolbox::cnst_sqrt;


using std::abs, std::max, std::min, std::log, std::exp, std::sqrt, std::log1p,
      std::isinf;
using boost::math::digamma, boost::math::trigamma, boost::math::polygamma;
// Root finding:
using boost::math::tools::bracket_and_solve_root,
      boost::math::tools::toms748_solve,
      boost::math::tools::eps_tolerance;
// Integration:
using boost::math::quadrature::tanh_sinh,
      boost::math::quadrature::exp_sinh,
      boost::math::quadrature::gauss_kronrod;

/*
 * Constructor:
 */
GammaConjugatePriorBase::GammaConjugatePriorBase(double lp, double s, double n,
                                                 double v, double amin,
                                                 double epsabs, double epsrel)
    : amin(amin), epsrel(epsrel), epsabs(epsabs), lp_(lp), p_(exp(lp)),
      ls(log(s)), s_(s), v_(v), n_(n),
      lPhi(ln_Phi(lp, ls, n, v, amin, epsabs, epsrel))
{
}


/*
 * Parameter access:
 */

double GammaConjugatePriorBase::lp() const
{
	return lp_;
}

double GammaConjugatePriorBase::p() const
{
	return p_;
}

double GammaConjugatePriorBase::s() const
{
	return s_;
}

double GammaConjugatePriorBase::n() const
{
	return n_;
}

double GammaConjugatePriorBase::v() const
{
	return v_;
}



template<typename real_t>
struct params0_t {
	real_t lp;
	real_t ls;
	real_t n;
	real_t v;
	real_t lFmax;
	double epsrel;
};

/* Computes the logarithm of the integrand of the normalization
 * constant Phi.
 */
template<typename real_t>
real_t ln_F(real_t a, real_t lp, real_t ls, real_t n, real_t v)
{
	return (a-1) * lp + std::lgamma(v*a) - n * std::lgamma(a)
	       - v * a * ls;
}


/* Compute the maximum of the integrand. */
template<typename real_t>
real_t amax_f0(real_t a, real_t v, real_t n)
{
	if (a > 0.1)
		return v*digamma(v*a) - n*digamma(a);
	/* For small values of a, use recurrence formula: */
	return v*digamma(v*a + 1) - n*digamma(a+1) + (n-1)/a;
}


template<typename real_t>
real_t amax_f1(real_t a, real_t v, real_t n)
{
	if (a > 1e8){
		/* Stirling's formula is accurate. */
		const real_t ia = 1/a;
		return (v-n)*ia + 0.5*(1-n)*ia*ia;
	}
	return v*v*trigamma(v*a) - n*trigamma(a);
}


template<typename real_t>
real_t amax_f2(real_t a, real_t v, real_t n)
{
	return v*v*v*polygamma(2,v*a) - n*polygamma(2,a);
}

template<typename real_t>
real_t compute_amax(const params0_t<real_t>& P, const real_t amin=1e-12)
{
	// Newton-Raphson from a low starting point:
	real_t f0,f1, da;
	const real_t C = P.lp - P.v * P.ls;
	real_t a = (amin < 1.0) ? 1.0 : amin;
	bool sign_min = amax_f0(amin, P.v, P.n) + C > 0.0;
	if (!sign_min)
		throw std::runtime_error("The derivative of the log-integrand is not "
		                         "positive at amin.");
	size_t i=0;
	real_t amax = a;
	while (amax_f0(amax, P.v, P.n) + C > 0.0 && i < 400){
		amax *= 100;
	}
	a = 0.5 * (amin + amax);

	for (i=0; i<200; ++i){
		f0 = amax_f0(a, P.v, P.n) + C;
		f1 = amax_f1(a, P.v, P.n);
		// Limit speed to exponential:
		da = min<real_t>(max<real_t>(-f0 / f1, -0.99*(a-amin)),
		                 0.99*(amax-a));
		a += da;
		a = max<real_t>(a, amin);
		if (std::fabs(da) <= 1e-12 * a)
			break;
	}

	return a;
}


template<typename real_t>
constexpr size_t log2flr(const real_t r){
	if (r > 9223372036854775808u)
		return log2flr(r/9223372036854775808u) + 100;
	if (r > 1048576.0)
		return log2flr(r/1048576) + 20;
	if (r > 1024.0)
		return log2flr(r/1024) + 10;
	if (r > 256.0)
		return log2flr(r/256) + 8;
	if (r > 16.0)
		return log2flr(r/16) + 4;
	if (r > 4.0)
		return log2flr(r/4) + 2;
	if (r >= 2.0)
		return log2flr(r/2) + 1;
	return 0;
}

/*
 * This method computes the maximum of the derivative of the integrand.
 */
template<typename real_t>
real_t derivative_amax(const real_t v, const real_t n)
{
	/* Use Newton-Raphson on the derivative of the derivative (second
	 * derivative of the log of the integrand) to determine the
	 * maximum of the derivative.
	 * Problem: The function 'amax_f1' has (for n < 1, where this function
	 * is called) one root and one asymptotic root at infinity - at least
	 * for some parameter combinations.
	 * Make sure that the Newton-Raphson iteration does not start in the
	 * attractor set of the latter. We do this by finding a point `ar` where
	 *   1) amax_f2 <= 0.0     and
	 *   2) amax_f1 <= 0.0
	 * Condition 2) ensures, together with the limit amax_f1 -> +infty for
	 * a -> 0, that `ar` is an upper bound of a monotonous bracket for the root.
	 */
	size_t iter = 0;
	real_t ar = 1.0;
	real_t f1_r;
	constexpr size_t imax = log2flr(std::numeric_limits<real_t>::max()) + 2;
	while (((f1_r = amax_f1(ar, v, n)) > 0.0) && (iter < imax)){
		ar *= 2;
	}
	/* It may happen that ar is outside the dynamic range of `real_t`: */
	if (f1_r > 0.0)
		return ar;

	while (amax_f2(ar, v, n) > 0.0 && iter < 10000){
		real_t ar_new = ar / 2;
		while (amax_f1(ar_new, v, n) > 0.0 && iter < 10000){
			ar_new = (ar + ar_new) / 2;
			++iter;
		}
		ar = ar_new;
		++iter;
	}
	real_t a0 = ar / 2;
	for (int i=0; i<100; ++i){
		const real_t f0 = amax_f1(a0, v, n);
		const real_t f1 = amax_f2(a0, v, n);
		const real_t a1 = min(max(a0 - f0/f1, 0.01*a0), ar - 0.01*(ar - a0));
		if (std::abs(a1-a0) < 1e-14 * a0){
			a0 = a1;
			break;
		}
		a0 = a1;
	}
	return a0;
}


template<typename real_t>
struct bounds_t {
	real_t l;
	real_t r;
};

/*
 * This function computes integration bounds within which the integrand's
 * mass is located primarily.
 */
template<typename real_t>
bounds_t<real_t>
peak_integration_bounds(const real_t amax, const double epsrel,
                        const real_t epsabs, const real_t f2_amax,
                        params0_t<real_t>& P)
{
	const real_t nleps0 = -log(epsrel) + log(1e4);
	const real_t leps = log(epsrel);
	const real_t da = sqrt(2/f2_amax * nleps0);
	const real_t lFmax = ln_F(amax, P.lp, P.ls, P.n, P.v);

	/* Left boundary: */
	std::uintmax_t iterations = 100;
	real_t al = max<real_t>(amax-da, 0.0);
	{
		auto rootfun = [&](real_t a) -> real_t {
			return (ln_F(a, P.lp, P.ls, P.n, P.v) - lFmax) - leps;
		};
		if (rootfun(0.0) < 0.0){
			if (al > 0.0 && rootfun(al) >= 0.0){
				std::pair<real_t,real_t> left
				   = toms748_solve(rootfun, static_cast<real_t>(0.0), al,
				                   eps_tolerance<real_t>(3), iterations);
				al = 0.5 * (left.first + left.second);
			} else {
				std::pair<real_t,real_t> left
				   = toms748_solve(rootfun, al, amax,
				                   eps_tolerance<real_t>(3), iterations);
				al = 0.5 * (left.first + left.second);
			}
		} else {
			al = 0.0;
		}
	}

	/* Right boundary: */
	real_t ar;
	iterations = 100;
	{
		auto rootfun = [&](real_t a) -> real_t {
			return (ln_F(a+amax, P.lp, P.ls, P.n, P.v) - lFmax) - leps;
		};

		try {
			std::pair<real_t,real_t> right
			   = bracket_and_solve_root(rootfun, da, static_cast<real_t>(2.0),
			                            false, eps_tolerance<real_t>(3),
			                            iterations);

			ar = amax + 0.5*(right.first + right.second);
		} catch (const std::runtime_error&){
			/* Set to infinity: */
			ar = std::numeric_limits<double>::infinity();
		}

	}

	return {.l=al, .r=ar};
}


static bool condition_warn(double S, double err, double L1, int where,
                           double critical_condition_number)
{
	const double condition_number = L1 / std::abs(S);
	bool do_warn = false;
	if (condition_number > critical_condition_number){
		std::cerr << "WARNING: Large condition number (" << condition_number
		          << ") in gamma conjugate prior "
		             "quadrature (in line " << where << ")\n" << std::flush;
		do_warn = true;
	}
	if (err > 1e-3 * std::abs(S)){
		std::cerr << "WARNING: Large relative error (" << err / std::abs(S)
		          << ") in gamma conjugate prior quadrature (in line " << where
		          << ")\n" << std::flush;
		do_warn = true;
	}
	return do_warn;
}



/*
 * Compute the natural logarithm of the integration constant:
 */

template<typename real_t>
real_t ln_Phi_backend(const real_t lp, const real_t ls, const real_t n,
                      const real_t v, const real_t amin, const real_t epsabs,
                      double epsrel)
{
	/* Shortcut for the case that p=0, in which case the
	 * integral is zero. This case is not particularly useful
	 * for the conjugate prior itself since then there is a zero
	 * in the denominator, but it can appear when evaluating
	 * the posterior predictive for q=0 (although that is not
	 * strictly allowed).
	 */
	if (std::isinf(lp) && lp < 0)
		return -std::numeric_limits<double>::infinity();

	/* `amin` sanity: */
	if (amin < 0)
		throw std::runtime_error("Parameter amin needs to be positive or "
		                         "zero.");

	/* Get a parameter structure: */
	params0_t<real_t> P = {.lp=lp, .ls=ls, .n=n, .v=v, .lFmax=0.0,
	                       .epsrel=epsrel};

	/* Determine the extremal structure of the integrand, one of three cases:
	 * 1) n > 1:
	 *    The derivative of the integrand is monotonously decreasing from
	 *    positive infinity at a=0. Hence, it has exactly one root.
	 *
	 * 2) n == 1:
	 *    The derivative of the integrand is finite at a=0 and monotonously
	 *    decreasing from there on. Hence, it has one root if the limit for
	 *    a --> 0 is positive or none if it is negative.
	 *
	 * 3) n < 1:
	 *    The derivative of the integrand has one maximum and goes to
	 *    negative infinity for a --> 0 and a --> inf.
	 *    Hence, the integrand has either no local extremum, a saddlepoint, or
	 *    a local minimum and a local maximum depending on whether the
	 *    integrand's derivative is negative, zero, or positive at its maximum.
	 */
	uint8_t n_extrema;
	real_t amax = 0.0;

	if (n > 1){
		/* Case 1, guaranteed extremum: */
		n_extrema = 1;
		amax = compute_amax(P);
	} else if (n == 1) {
		/* Case 2, have to check derivative at a=0: */
		constexpr real_t euler_mascheroni = 0.577215664901532860606512090082;
		if (lp - v*ls + (1-v)*euler_mascheroni > 0){
			n_extrema = 1;
			amax = compute_amax(P);
		} else {
			n_extrema = 0;
		}
	} else {
		/* Case 3 */
		const real_t amax_deriv = derivative_amax(v, n);
		if (amax_f0<real_t>(amax_deriv, v, n) + lp - v*ls > 0){
			/* We have a second maximum (local extremum), which is right of the
			 * maximum of the derivative: */
			n_extrema = 2;
			amax = compute_amax(P, amax_deriv);
		} else {
			/* Either no extremum or a saddle point - both can be handled
			 * the same way. */
			n_extrema = 0;
		}
	}

	/* If amax is outside the numerical range, the normalization constant
	 * will be as well.
	 * Return infinity; the posterior cannot be resolved.
	 */
	if (std::isinf(static_cast<double>(amax)))
		throw std::runtime_error("Cannot normalize the integration constant"
		                         " since its maximum `amax` cannot be "
		                         "expressed in double precision.");

	/* Now depending on the extremal structure, we have to choose different
	 * integration techniques: */
	real_t err, L1;
	real_t res = 0.0 ;
	constexpr double term = cnst_sqrt(std::numeric_limits<double>::epsilon());

	auto integrand1 = [&](real_t a, real_t distance_to_next_bound) -> real_t
	{
		const real_t res
		   = exp(ln_F(a, P.lp, P.ls, P.n, P.v) - P.lFmax);
		if (isinf(res))
			std::cerr << "Found inf result in integrand for amax = "
			          << amax << ", a = " << a << " (integrand 1).\n";
		return res;
	};

	if (n_extrema >= 1 && amax > amin){
		/* Compute the maximum value of the logarithm of the integrand: */
		P.lFmax = ln_F<real_t>(amax, lp, ls, n, v);

		/*
		 * A Laplace approximation for very large amax:
		 */
		if (amax > 1e10){
			/* Second derivative of the logarithm of the integrand at the maximum,
			 * i.e. the value used for a Laplace approximation of the integral:
			 */
			const real_t f2_amax
			    = std::fabs(amax_f1<real_t>(amax, v, n));

			/* Compute the upper and lower boundary of integration, i.e. the
			 * points where the integrand diminishes to machine precision:
			 */
			bounds_t<real_t> bounds
			   = peak_integration_bounds<real_t>(amax, epsrel, epsabs,
			                                     f2_amax, P);

			/* For consistency checking, make sure that the first non-quadratic
			 * term of the Taylor expansion of the log-integrand (the cubic
			 * term) is less than 4.61 (=log(1e2)) at the location of the
			 * cutoff: */
			const real_t lf3 = -2 * log(amax)
			                        + log(abs(n*(1+1/amax)
			                              - v*(1+1/(v*amax))));

			if (exp(lf3 - 3*log(max(bounds.l, bounds.r))) > 6.0 * 4.61)
				throw std::runtime_error("Potential roundoff error detected in "
				                         "Laplace approximation for "
				                         "a_max > 1e10.");

			return 0.5 * log(2*(real_t)M_PI) - 0.5 * log(f2_amax)
			       + P.lFmax;
		}

		/* Split the integral into two parts:
		 * (amin, amax) -> use tanh-sinh quadrature
		 * (amax,  inf) -> use exp-sinh quadrature
		 *
		 * (1) From 0 to amax:
		 */
		real_t S = 0.0;
		{
			tanh_sinh<real_t> ts;
			const real_t I = ts.integrate(integrand1, amin, amax, term,
			                                   &err, &L1);
			S += I;
			condition_warn(I, err, L1, __LINE__, 3.0);
		}

		/* (2) From amax to inf: */
		{
			auto integrand2 = [&](real_t a) -> real_t {
				const real_t res
				   = exp(ln_F(a+amax, P.lp, P.ls, P.n, P.v)
				          - P.lFmax);
				if (std::isinf(res)){
					std::cerr << "Found inf result in integrand2 for amax = "
					          << amax << ", a = " << a << ".\n";
					std::cerr << "P.lFmax         = " << P.lFmax << "\n";
					std::cerr << "ln_F(a)         = " << ln_F(a+amax, P.lp,
					                                          P.ls, P.n, P.v)
					          << "\n";
					std::cerr << "ln_F(a) - lFmax = " << ln_F(a+amax, P.lp,
					                                          P.ls, P.n, P.v)
					          << "\n";
				}
				return res;
			};
			exp_sinh<real_t> es;
			const real_t I = es.integrate(integrand2, term, &err, &L1);
			S += I;
			condition_warn(I, err, L1, __LINE__, 3.0);
		}
		return P.lFmax + log(S);
	} else {
		/* Integrate the singularity at a=0 and for the rest, integrate to
		 * infinity: */
		P.lFmax = ln_F(max(amin, (real_t)1.0), lp, ls, n, v);
		res = P.lFmax;

		auto integrand3 = [&](real_t a) -> real_t {
			const real_t res
			   = exp(ln_F(a+amin, P.lp, P.ls, P.n, P.v) - P.lFmax);
			if (isinf(res)){
				std::stringstream buf;
				buf << "Found inf result in integrand for amax = ";
				buf << std::setprecision(16);
				buf << amax;
				buf << ", a = ";
				buf << a;
				buf << " (integrand 3).\n  lF =    ";
				buf << ln_F(a+amin, P.lp, P.ls, P.n, P.v);
				buf << "\n  lFmax = ";
				buf << P.lFmax;
				buf << "\n  lp = ";
				buf << P.lp;
				buf << "\n  ls = ";
				buf << P.ls;
				buf << "\n  n =  ";
				buf << P.n;
				buf << "\n  v = ";
				buf << P.v;
				buf << "\n";
				buf << "  n_extrema = " << (int)n_extrema << "\n";
				std::cerr << buf.str();
			}
			return res;
		};

		exp_sinh<real_t> es;
		const real_t I = es.integrate(integrand3, term, &err, &L1);
		condition_warn(I, err, L1, __LINE__, 3.0);
		res += log(I);
	}

	/* Result: */
	return res;
}


double GammaConjugatePriorBase::ln_Phi(double lp, double ls, double n, double v,
                                       double amin, double epsabs,
                                       double epsrel)
{
	/* Return: */
	return ln_Phi_backend(lp, ls, n, v, amin, epsabs, epsrel);
}



/*
 * Kullback-Leibler distance:
 */
static double kullback_leibler_base(
    const double lp, const double s, const double ls, const double n,
    const double v, const double lp_ref, const double s_ref,
    const double ls_ref, const double n_ref, const double v_ref,
    const double amin, const long double ln_Phi, const long double ln_Phi_ref,
    double epsabs, double epsrel,
    pdtoolbox::condition_policy_t on_condition_large
)
{
	/* The integral: */
	auto integrand = [=](long double a) -> long double {
		a += amin;
		const long double lga = lgamma(a);
		const long double C0 = lp_ref - lp;
		const long double C3 = v - v_ref;
		const long double C1 = -C0 - v * (s - s_ref) / s
		                      - ls * C3;
		const long double C2 = - (n - n_ref);
		return exp((a - 1.0)*lp - n*lga + lgamma(a*v) - v * a * ls
		           - ln_Phi)
		       * ((C1 + C3 * digamma(a*v)) * a  +  C0  +  C2 * lga);
	};

	exp_sinh<long double> es;
	constexpr double term = cnst_sqrt(std::numeric_limits<double>::epsilon());
	long double err, L1;
	const long double I = es.integrate(integrand, term, &err, &L1);

	/* The integrand changes sign due to non-exponentiated part.
	 * A moderately large condition number (~1e3) occurs rather frequently
	 * without the integrand being highly oscillatory (typically
	 * perhaps one change of sign).
	 * Warn for a condition number of 1e5 - this should still give
	 * sufficient precision from the remaining ~10 digits.
	 */
	switch (on_condition_large){
		case pdtoolbox::CONDITION_WARN:
			condition_warn(I, err, L1, __LINE__, 1e5);
			break;
		case pdtoolbox::CONDITION_ERROR:
			if (err * 1e5 > L1){
				throw std::runtime_error("Large condition number in "
				                         "GammaConjugatePriorBase::"
				                         "kullback_leibler");
			}
			break;
		case pdtoolbox::CONDITION_INF:
			if (err * 1e5 > L1){
				return std::numeric_limits<double>::infinity();
			}
			break;
	}

	return ln_Phi_ref - ln_Phi + I;
}



double GammaConjugatePriorBase::kullback_leibler(
            double lp, double s, double n, double v,
            double lp_ref, double s_ref, double n_ref,
            double v_ref, double amin, double epsabs, double epsrel,
            condition_policy_t on_condition_large)
{
	const double ls = log(s);
	const double ls_ref = log(s_ref);
	const long double ln_Phi = ln_Phi_backend<long double>(lp, ls, n, v, amin,
	                                                       epsabs, epsrel);
	const long double ln_Phi_ref = ln_Phi_backend<long double>(lp_ref, ls_ref,
	                                                           n_ref, v_ref,
	                                                           amin, epsabs,
	                                                           epsrel);

	return kullback_leibler_base(lp, s, ls, n, v, lp_ref, s_ref, ls_ref,
	                             n_ref, v_ref, amin, ln_Phi, ln_Phi_ref,
	                             epsabs, epsrel, on_condition_large);
}


double GammaConjugatePriorBase::kullback_leibler_batch_max(
            const double* params, size_t N,
            double lp_ref, double s_ref, double n_ref, double v_ref,
            double amin, double epsabs, double epsrel,
            const long double* ln_Phis)
{
	/* First determine the normalization constant: */
	const double ls_ref = log(s_ref);
	long double ln_Phi_ref;
	try {
		ln_Phi_ref
		   = ln_Phi_backend<long double>(lp_ref, ls_ref, n_ref, v_ref,
		                                 amin, epsabs, epsrel);
	} catch (...) {
		return std::numeric_limits<double>::infinity();
	}

	std::vector<double> KL(N);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		const double lp = params[4*i];
		const double s  = params[4*i+1];
		const double n  = params[4*i+2];
		const double v  = params[4*i+3];
		const double ls = log(s);

		try {
			const long double ln_Phi
			   = (ln_Phis) ? ln_Phis[i]
			               : ln_Phi_backend<long double>(lp, ls, n, v, amin,
			                                             epsabs, epsrel);

			KL[i] = kullback_leibler_base(lp, s, ls, n, v, lp_ref, s_ref,
			                              ls_ref, n_ref, v_ref, amin, ln_Phi,
			                              ln_Phi_ref, epsabs, epsrel,
			                              CONDITION_ERROR);

		} catch (...) {
			/* On error, reduce to maximum: */
			KL[i] = std::numeric_limits<double>::infinity();
		}
	}

	return *std::max_element(KL.cbegin(), KL.cend());
}


std::shared_ptr<pdtoolbox::SortedCache<4,double>>
GammaConjugatePriorBase::generate_mse_cost_function_cache(
    const double* params, size_t N, double amin, double epsabs, double epsrel
)
{
	/* Compute the normalization constants: */
	std::vector<long double> ln_Phi(N);
	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		ln_Phi[i] = ln_Phi_backend<long double>(params[4*i], log(params[4*i+1]),
		                                        params[4*i+2], params[4*i+3],
		                                        amin, epsabs, epsrel);
	}

	/* The lambda capture: */
	auto fun = [=](const std::array<double,4>& arg) -> double {
		// Compute:
		return kullback_leibler_batch_max(params, N, arg[0], arg[1], arg[2],
		                                  arg[3], amin, epsabs, epsrel,
		                                  ln_Phi.data());
	};

	/* The cache: */
	typedef SortedCache<4,double> cache_t;
	std::shared_ptr<cache_t> cache
	   = std::make_shared<cache_t>(fun);

	return cache;
}


std::shared_ptr<pdtoolbox::SortedCache<4,double>>
GammaConjugatePriorBase::restore_mse_cost_function_cache(
    const double* cache_dump, size_t M, const double* params, size_t N,
    double amin, double epsabs, double epsrel
)
{
	/* Load the cache dump: */
	if ((M % 5) != 0)
		throw std::runtime_error("Size of cache dump array is wrong.");

	const size_t m = M / 5;
	std::vector<std::pair<const std::array<double,4>, double>> structured;
	structured.reserve(m);
	for (size_t i=0; i<m; ++i){
		const std::array<double,4> key({cache_dump[5*i], cache_dump[5*i+1],
		                                cache_dump[5*i+2], cache_dump[5*i+3]});
		structured.push_back(std::make_pair(key, cache_dump[5*i+4]));
	}

	/* Compute the normalization constants: */
	std::vector<long double> ln_Phi(N);
	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		ln_Phi[i] = ln_Phi_backend<long double>(params[4*i], log(params[4*i+1]),
		                                        params[4*i+2], params[4*i+3],
		                                        amin, epsabs, epsrel);
	}

	/* The lambda capture: */
	auto fun = [=](const std::array<double,4>& arg) -> double {
		// Compute:
		return kullback_leibler_batch_max(params, N, arg[0], arg[1], arg[2],
		                                  arg[3], amin, epsabs, epsrel,
		                                  ln_Phi.data());
	};

	/* The cache: */
	typedef SortedCache<4,double> cache_t;
	std::shared_ptr<cache_t> cache
	   = std::make_shared<cache_t>(fun, structured);

	return cache;
}

/*
 * Posterior predictive PDF:
 */

void GammaConjugatePriorBase::posterior_predictive_pdf(
        const size_t Nq, const double* q, double* out,
        double lp, double s, double n, double v,
        double amin, double epsabs, double epsrel, bool parallel)
{
	if (Nq == 0)
		return;

	/* The normalization: */
	const double ls = std::log(s);
	const double lnPhi = ln_Phi(lp, ls, n, v, amin, epsabs, epsrel);

	/* Call the backend: */
	posterior_predictive_pdf(Nq, q, out, lp, s, n, v, amin, epsabs, epsrel,
	                         parallel, lnPhi, ls);
}

void GammaConjugatePriorBase::posterior_predictive_pdf(
        const size_t Nq, const double* q, double* out,
        double lp, double s, double n, double v,
        double amin, double epsabs, double epsrel, bool parallel,
        double lnPhi, double ls)
{
	if (Nq == 0)
		return;

	/* Error catching: */
	bool error_flag = false;
	std::string err_msg;

	#pragma omp parallel for if(parallel)
	for (size_t i=0; i<Nq; ++i){
		/* Although not strictly defined, be gracious and
		 * assign p(q<=0) = 0.0
		 */
		if (q[i] <= 0)
			out[i] = 0.0;
		else {
			try {
				out[i] = exp(ln_Phi(lp + log(q[i]), log(s+q[i]), n+1, v+1, amin,
				                    epsabs, epsrel)
				             - lnPhi);
			} catch (const std::exception& e) {
				error_flag = true;
				err_msg = std::string(e.what());
			}
		}
	}

	if (error_flag){
		err_msg = std::string("Error in posterior_predictive_pdf: '")
		          + err_msg + std::string("'.");
		throw std::runtime_error(err_msg);
	}
}

void GammaConjugatePriorBase::posterior_predictive_pdf_batch(
        const size_t Nq, const double* q, double* out,
        const size_t Mparam, const double* lp, const double* s,
        const double* n, const double* v, double amin,
        double epsabs, double epsrel, bool parallel)
{
	if (Nq == 0)
		return;

	/* Error catching: */
	bool error_flag = false;
	std::string err_msg;

	#pragma omp parallel for if(parallel)
	for (size_t i=0; i<Mparam; ++i){
		try {
			posterior_predictive_pdf(Nq, q, out + Nq*i, lp[i], s[i], n[i], v[i],
			                         amin, epsabs, epsrel, false);
		} catch (const std::exception& e){
			error_flag = true;
			err_msg = std::string(e.what());
		}
	}

	if (error_flag){
		err_msg = std::string("Error in posterior_predictive_pdf_batch: '")
		          + err_msg + std::string("'.");
		throw std::runtime_error(err_msg);
	}
}



/*
 * Posterior predictive CDF:
 */
void GammaConjugatePriorBase::posterior_predictive_cdf(
        const size_t Nq, const double* q, double* out,
        double lp, double s, double n, double v,
        double amin, double epsabs, double epsrel
)
{
	if (Nq == 0)
		return;

	/* Sort the heat flow values: */
	struct orddbl {
		double val = 0;
		size_t index = 0;
		bool operator<(const orddbl& other) const {
			return val < other.val;
		};
	};
	std::vector<orddbl> qord(Nq);
	for (size_t i=0; i<Nq; ++i){
		qord[i] = {q[i], i};
	}
	std::sort(qord.begin(), qord.end());

	/* Reference Phi: */
	const double ls = log(s);
	const double ln_Phi = ln_Phi_backend(lp, ls, n, v, amin, epsabs, epsrel);

	/* The integrand: */
	auto integrand = [=](double qi) -> double {
		return exp(ln_Phi_backend(lp + log(qi), log(s + qi), n+1, v+1, amin,
		                          epsabs, epsrel)
		           - ln_Phi);
	};

	/* Now the piecewise integration.
	 * This loop could be parallelized easily: */
	double err;
	for (size_t i=0; i<Nq; ++i){
		/* Numerical integration of this part: */
		const size_t j = qord[i].index;
		double ql = (i == 0) ? 0.0 : qord[i-1].val;
		if (qord[i].val == ql)
			out[j] = 0.0;
		else
			out[j] = gauss_kronrod<double, 15>::integrate(integrand,
			                                              ql, qord[i].val,
			                                              5, 1e-9, &err);

		/* Error checking? */
	}

	/* Accumulation: */
	for (size_t i=1; i<Nq; ++i){
		out[qord[i].index] += out[qord[i-1].index];
	}

	/* Ensure normalization: */
	const double Send = out[qord[Nq - 1].index];
	if (Send > 1.0){
		for (size_t i=0; i<Nq; ++i){
			out[i] /= Send;
		}
	}
}


void
GammaConjugatePriorBase::posterior_predictive_cdf_batch(
        const size_t Nq, const double* q, double* out, const size_t Mparam,
        const double* lp, const double* s, const double* n, const double* v,
        double amin, double epsabs, double epsrel)
{
	bool error_flag = false;
	std::string err_msg;

	#pragma omp parallel for
	for (size_t i=0; i<Mparam; ++i){
		try {
			posterior_predictive_cdf(Nq, q, out + Nq*i, lp[i], s[i], n[i], v[i],
			                         amin, epsabs, epsrel);
		} catch (const std::exception& e){
			error_flag = true;
			err_msg = std::string(e.what());
		}
	}

	if (error_flag){
		err_msg = std::string("Error in posterior_predictive_cdf_batch: '")
		          + err_msg + std::string("'.");
		throw std::runtime_error(err_msg);
	}
}
