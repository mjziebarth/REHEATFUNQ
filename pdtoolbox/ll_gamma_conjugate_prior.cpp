/*
 * Conjugate prior of the gamma distribution due to Miller (1980).
 *
 * Copyright (C) 2021 Deutsches GeoForschungsZentrum GFZ
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

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <ll_gamma_conjugate_prior.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <optimize.hpp>

#include <iostream>
#include <iomanip>


/* Namespae: */
using namespace pdtoolbox;
using std::abs, std::max, std::min, std::log, std::exp, std::sqrt;
using boost::math::digamma, boost::math::trigamma, boost::math::polygamma;
// Root finding:
using boost::math::tools::bracket_and_solve_root,
      boost::math::tools::toms748_solve,
      boost::math::tools::eps_tolerance;
// Integration:
using boost::math::quadrature::tanh_sinh,
      boost::math::quadrature::exp_sinh,
      boost::math::quadrature::gauss_kronrod;


static_assert(GCP_AMIN >= 0.0);


struct params0_t {
	double lp;
	double ls;
	double n;
	double v;
	double lFmax;
	double epsrel;
};

/* Computes the logarithm of the integrand of the normalization
 * constant Phi.
 */
static double ln_F(double a, double lp, double ls, double n, double v)
{
	return (a-1) * lp + std::lgamma(v*a) - n * std::lgamma(a)
	       - v * a * ls;
}


/* Compute the maximum of the integrand. */
static double amax_f0(double a, double v, double n)
{
	if (a > 0.1)
		return v*digamma(v*a) - n*digamma(a);
	/* For small values of a, use recurrence formula: */
	return v*digamma(v*a + 1) - n*digamma(a+1) + (n-1)/a;
}


static double amax_f1(double a, double v, double n)
{
	if (a > 1e8){
		/* Stirling's formula is accurate. */
		const double ia = 1/a;
		return (v-n)*ia + 0.5*(1-n)*ia*ia;
	}
	return v*v*trigamma(v*a) - n*trigamma(a);
}


static double amax_f2(double a, double v, double n)
{
	return v*v*v*polygamma(2,v*a) - n*polygamma(2,a);
}


static double compute_amax(const params0_t& P, const double amin=1e-12)
{
	// Initial guess from Stirling's formula expansion:
	double a = std::max(std::exp(
	                     (P.lp - P.v*P.ls + P.v*std::log(P.v)) / (P.n-P.v)),
	                    std::max(1.0, 1.1*amin));

	// If we are above a=1e8, Stirling's formula is accurate to machine
	// precision
	if (a >= 1e8)
		return a;

	// Newton-Raphson to improve:
	double f0,f1, da;
	const double C = P.lp - P.v * P.ls;

	for (size_t i=0; i<200; ++i){
		f0 = amax_f0(a, P.v, P.n) + C;
		f1 = amax_f1(a, P.v, P.n);
		da = f0 / f1;
		a -= da;
		a = std::max(a, amin);
		if (std::fabs(da) <= 1e-12 * a)
			break;
	}

	return a;
}


/*
 * This method computes the maximum of the derivative of the integrand.
 */
static double derivative_amax(double v, double n)
{
	// Use Newton-Raphson on the derivative of the derivative (second
	// derivative of the log of the integrand) to determine the
	// maximum of the derivative:
	double a0 = 1.0;
	for (int i=0; i<100; ++i){
		const double f0 = amax_f1(a0, v, n);
		const double f1 = amax_f2(a0, v, n);
		const double a1 = std::max(a0 - f0/f1, 1e-12);
		if (std::abs(a1-a0) < 1e-14 * a0){
			a0 = a1;
			break;
		}
		a0 = a1;
	}
	return a0;
}


struct bounds_t {
	double l;
	double r;
};

/*
 * This function computes integration bounds within which the integrand's
 * mass is located primarily.
 */
static bounds_t
peak_integration_bounds(const double amax, const double epsrel,
                        const double epsabs, const double f2_amax, params0_t& P)
{
	const double nleps0 = -log(epsrel) + log(1e4);
	const double leps = log(epsrel);
	const double da = sqrt(2/f2_amax * nleps0);
	const double lFmax = ln_F(amax, P.lp, P.ls, P.n, P.v);

	/* Left boundary: */
	std::uintmax_t iterations = 100;
	double al = max(amax-da, 0.0);
	{
		auto rootfun = [&](double a) -> double {
			return (ln_F(a, P.lp, P.ls, P.n, P.v) - lFmax) - leps;
		};
		if (rootfun(0.0) < 0.0){
			if (al > 0.0 && rootfun(al) >= 0.0){
				std::pair<double,double> left
				   = toms748_solve(rootfun, 0.0, al,
				                   eps_tolerance<double>(3), iterations);
				al = 0.5 * (left.first + left.second);
			} else {
				std::pair<double,double> left
				   = toms748_solve(rootfun, al, amax,
				                   eps_tolerance<double>(3), iterations);
				al = 0.5 * (left.first + left.second);
			}
		} else {
			al = 0.0;
		}
	}

	/* Right boundary: */
	double ar;
	iterations = 100;
	{
		auto rootfun = [&](double a) -> double {
			return (ln_F(a+amax, P.lp, P.ls, P.n, P.v) - lFmax) - leps;
		};

		try {
			std::pair<double,double> right
			   = bracket_and_solve_root(rootfun, da, 2.0, false,
			                            eps_tolerance<double>(3), iterations);

			ar = amax + 0.5*(right.first + right.second);
		} catch (const std::runtime_error&){
			/* Set to infinity: */
			ar = std::numeric_limits<double>::infinity();
		}

	}

	return {.l=al, .r=ar};
}

/* Returns the logarithm of the sum of two numbers given as logarithms. */
static double add_logs(double la, double lb)
{
	constexpr double leps = log(1e-16);
	double lmin = min(la,lb);
	double lmax = max(la,lb);
	if (lmin < lmax + leps){
		return lmax;
	}
	return lmax + std::log1p(exp(lmin-lmax));
}


static void condition_warn(double S, double err, double L1)
{
	const double condition_number = L1 / std::abs(S);
	if (condition_number > 3){
		std::cerr << "WARNING: Large condition number in gamma conjugate prior "
		             "quadrature.\n" << std::flush;
	}
	if (err > 1e-3 * std::abs(S)){
		std::cerr << "WARNING: Large relative error (" << err / std::abs(S)
		          << ") in gamma conjugate prior quadrature.\n" << std::flush;
	}
}


/*
 * Compute the natural logarithm of the integration constant:
 */
//#define GAMMA_CONJUGATE_PRIOR_OLD_LN_PHI
#ifdef GAMMA_CONJUGATE_PRIOR_OLD_LN_PHI

static double ln_Phi_backend(double lp, double ls, double n, double v,
                             double epsabs, double epsrel)
{
	static_assert(false, "Implement GCP_AMIN!");

	/* Shortcut for the case that p=0, in which case the
	 * integral is zero. This case is not particularly useful
	 * for the conjugate prior itself since then there is a zero
	 * in the denominator, but it can appear when evaluating
	 * the posterior predictive for q=0 (although that is not
	 * strictly allowed).
	 */
	if (std::isinf(lp) && lp < 0)
		return -std::numeric_limits<double>::infinity();

	/* Get a parameter structure: */
	params0_t P = {.lp=lp, .ls=ls, .n=n, .v=v, .lFmax=0.0, .epsrel=epsrel};

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
	double amax = 0;

	if (n > 1){
		/* Case 1, guaranteed extremum: */
		n_extrema = 1;
		amax = compute_amax(P);
	} else if (n == 1) {
		/* Case 2, have to check derivative at a=0: */
		constexpr double euler_mascheroni = 0.577215664901532860606512090082;
		if (lp - v*ls + (1-v)*euler_mascheroni > 0){
			n_extrema = 1;
			amax = compute_amax(P);
		} else {
			n_extrema = 0;
		}
	} else {
		/* Case 3 */
		const double amax_deriv = derivative_amax(v, n);
		if (amax_f0(amax_deriv, v, n) + lp - v*ls > 0){
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

	/* Now depending on the extremal structure, we have to choose different
	 * integration techniques: */
	double S=0, err, L1;
	double res = 0.0 ;
	constexpr double term = std::sqrt(std::numeric_limits<double>::epsilon());

	auto integrand = [&](double a) -> double {
		double lF = ln_F(a, P.lp, P.ls, P.n, P.v);
		return std::exp(lF - P.lFmax);
	};

	if (n_extrema >= 1 && amax > 0){
		/* Compute the maximum value of the logarithm of the integrand: */
		if (std::isinf(amax))
			throw std::runtime_error("Cannot normalize the integration constant"
			                         " since its maximum `amax` cannot be "
			                         "expressed in double precision.");
		P.lFmax = ln_F(amax, lp, ls, n, v);
		res = P.lFmax;

		/* Second derivative of the logarithm of the integrand at the maximum,
		 * i.e. the value used for a Laplace approximation of the integral:
		 */
		const double f2_amax = std::fabs(amax_f1(amax, v, n));


		/* Compute the upper and lower boundary of integration, i.e. the points
		 * where the integrand diminishes to machine precision:
		 */
		bounds_t bounds = peak_integration_bounds(amax, epsrel, epsabs, f2_amax,
		                                          P);


		/* If we have two extrema, integrate the lower part: */
		if (n_extrema == 2){
			/* First we need to make sure that the lower bound is actually
			 * bigger than 0: */
			 if (bounds.l == 0.0)
				 bounds.l = min(1.0, 0.5 * amax);

			/* Now integrate the divergent integrand: */
			tanh_sinh<double> ts;
			const double I = ts.integrate(integrand, 0.0, bounds.l, term,
			                              &err, &L1);
			S += I;
			condition_warn(I, err, L1);
		}

		/* Check if the bounds are finite: */
		if (std::isinf(bounds.r)){
			/* In case of infinite bounds, use exp-sinh quadrature to
			 * integrate from the left boundary:
			 */
			auto integrand2 = [&](double a) -> double {
				double lF = ln_F(a+bounds.l, P.lp, P.ls, P.n, P.v);
				double itg = std::exp(lF - P.lFmax);
				if (std::isnan(itg))
					std::cerr << "NaN integrand:\n"
					             "   a = " << a << "\n"
					             "   lp= " << lp << "\n"
					             "   ls= " << ls << "\n"
					             "   n = " << n << "\n"
					             "   v = " << v << "\n";
				return itg;
			};
			exp_sinh<double> es;
			const double I = es.integrate(integrand2, term, &err, &L1);
			S += I;
			condition_warn(I, err, L1);
			res += std::log(S);
			return res;
		}

		/* A shortcut to Laplace approximation for very large amax: */
		const double da_l = amax - bounds.l, da_r = bounds.r - amax;
		if (amax > 1e10){
			/* For consistency checking, make sure that the first non-quadratic
			 * term of the Taylor expansion of the log-integrand (the cubic
			 * term) is less than 4.61 (=log(1e2)) at the location of the
			 * cutoff: */
			const double lf3 = -2 * log(amax) + log(abs(n*(1+1/amax)
			                                            - v*(1+1/(v*amax))));

			if (exp(lf3 - 3*log(max(bounds.l, bounds.r))) > 6.0 * 4.61)
				throw std::runtime_error("Potential roundoff error detected in "
				                         "Laplace approximation for "
				                         "a_max > 1e10.");
			return add_logs(0.5 * log(2*M_PI) - 0.5 * log(f2_amax),
			                log(S)) + P.lFmax;
		}

		/* Check if within this significant interval, the Laplace method is
		 * already a well-enough approximation:
		 * (we consider this to be the case if the difference between the
		 * quadratic approximation and the actual log-integrand is less than
		 * log(1e2) = 4.61.
		 * With this choice and our above enforcement of being 1e-20 of the
		 * integrand maximum, we are at a precision of 1e-18 compared to the
		 * integrand's maximum)
		 */
		const double dal2 = da_l * da_l,  dar2 = da_r * da_r;

		if (max(abs(ln_F(bounds.l,lp,ls,n,v) - P.lFmax + 0.5 * dal2 * f2_amax),
		        abs(ln_F(bounds.r,lp,ls,n,v) - P.lFmax + 0.5 * dar2 * f2_amax))
		    < 4.61)
		{
			/* Use the Laplace method: */
			if (S != 0){
				return add_logs(0.5 * log(2*M_PI) - 0.5 * log(f2_amax),
				                log(S)) + P.lFmax;
			}
			return 0.5 * log(2*M_PI) - 0.5 * log(f2_amax) + P.lFmax;
		}

		/* Otherwise numerically evaluate the integral: */
		S += gauss_kronrod<double, 15>::integrate(integrand, bounds.l, bounds.r,
		                                          5, 1e-14, &err);
		res += std::log(S);
	} else {
		/* Integrate the singularity at a=0 and for the rest, integrate to
		 * infinity: */
		P.lFmax = ln_F(1.0, lp, ls, n, v);
		res = P.lFmax;

		exp_sinh<double> es;
		const double I = es.integrate(integrand, term, &err, &L1);
		S += I;
		condition_warn(I, err, L1);
		res += std::log(S);
	}

	/* Result: */
	return res;
}

#else

/* Compute the natural logarithm of the integration constant: */
static double ln_Phi_backend(double lp, double ls, double n, double v,
                             double epsabs, double epsrel)
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

	/* Get a parameter structure: */
	params0_t P = {.lp=lp, .ls=ls, .n=n, .v=v, .lFmax=0.0, .epsrel=epsrel};

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
	double amax = 0.0;

	if (n > 1){
		/* Case 1, guaranteed extremum: */
		n_extrema = 1;
		amax = compute_amax(P);
	} else if (n == 1) {
		/* Case 2, have to check derivative at a=0: */
		constexpr double euler_mascheroni = 0.577215664901532860606512090082;
		if (lp - v*ls + (1-v)*euler_mascheroni > 0){
			n_extrema = 1;
			amax = compute_amax(P);
		} else {
			n_extrema = 0;
		}
	} else {
		/* Case 3 */
		const double amax_deriv = derivative_amax(v, n);
		if (amax_f0(amax_deriv, v, n) + lp - v*ls > 0){
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

	/* Now depending on the extremal structure, we have to choose different
	 * integration techniques: */
	double err, L1;
	double res = 0.0 ;
	constexpr double term = std::sqrt(std::numeric_limits<double>::epsilon());

	auto integrand1 = [&](double a) -> double {
		double lF = ln_F(a, P.lp, P.ls, P.n, P.v);
		if (std::isinf(std::exp(lF - P.lFmax)))
			std::cerr << "Found inf result in integrand for amax = "
			          << amax << ", a = " << a << ".\n";
		return std::exp(lF - P.lFmax);
	};

	if (n_extrema >= 1 && amax > GCP_AMIN){
		/* Compute the maximum value of the logarithm of the integrand: */
		if (std::isinf(amax))
			throw std::runtime_error("Cannot normalize the integration constant"
			                         " since its maximum `amax` cannot be "
			                         "expressed in double precision.");
		P.lFmax = ln_F(amax, lp, ls, n, v);


		/*
		 * A Laplace approximation for very large amax:
		 */
		if (amax > 1e10){
			/* Second derivative of the logarithm of the integrand at the maximum,
			 * i.e. the value used for a Laplace approximation of the integral:
			 */
			const double f2_amax = std::fabs(amax_f1(amax, v, n));

			/* Compute the upper and lower boundary of integration, i.e. the points
			 * where the integrand diminishes to machine precision:
			 */
			bounds_t bounds = peak_integration_bounds(amax, epsrel, epsabs,
			                                          f2_amax, P);

			const double da_l = amax - bounds.l, da_r = bounds.r - amax;
			/* For consistency checking, make sure that the first non-quadratic
			 * term of the Taylor expansion of the log-integrand (the cubic
			 * term) is less than 4.61 (=log(1e2)) at the location of the
			 * cutoff: */
			const double lf3 = -2 * log(amax) + log(abs(n*(1+1/amax)
			                                            - v*(1+1/(v*amax))));

			if (exp(lf3 - 3*log(max(bounds.l, bounds.r))) > 6.0 * 4.61)
				throw std::runtime_error("Potential roundoff error detected in "
				                         "Laplace approximation for "
				                         "a_max > 1e10.");

			return 0.5 * log(2*M_PI) - 0.5 * log(f2_amax) + P.lFmax;
			//return add_logs(0.5 * log(2*M_PI) - 0.5 * log(f2_amax),
			//                log(S)) + P.lFmax;
		}

		/* Split the integral into two parts:
		 * (amin, amax) -> use tanh-sinh quadrature
		 * (amax,  inf) -> use exp-sinh quadrature
		 *
		 * (1) From 0 to amax:
		 */
		double S = 0.0;
		{
			tanh_sinh<double> ts;
			const double I = ts.integrate(integrand1, GCP_AMIN, amax, term,
			                              &err, &L1);
			S += I;
			condition_warn(I, err, L1);
		}

		/* (2) From amax to inf: */
		{
			auto integrand2 = [&](double a) -> double {
				res = std::exp(ln_F(a+amax, P.lp, P.ls, P.n, P.v)
				                - P.lFmax);
				if (std::isinf(res)){
					std::cerr << "Found inf result in integrand2 for amax = "
					          << amax << ", a = " << a << ".\n";
					std::cerr << "P.lFmax         = " << P.lFmax << "\n";
					std::cerr << "ln_F(a)         = " << ln_F(a+amax, P.lp, P.ls, P.n, P.v) << "\n";
					std::cerr << "ln_F(a) - lFmax = " << ln_F(a+amax, P.lp, P.ls, P.n, P.v) << "\n";
				}
				return std::exp(ln_F(a+amax, P.lp, P.ls, P.n, P.v)
				                - P.lFmax);
			};
			exp_sinh<double> es;
			const double I = es.integrate(integrand2, term, &err, &L1);
			S += I;
			condition_warn(I, err, L1);
		}
		return P.lFmax + std::log(S);
	} else {
		/* Integrate the singularity at a=0 and for the rest, integrate to
		 * infinity: */
		P.lFmax = ln_F(max(GCP_AMIN, 1.0), lp, ls, n, v);
		res = P.lFmax;

		auto integrand3 = [&](double a) -> double {
			double lF = ln_F(a+GCP_AMIN, P.lp, P.ls, P.n, P.v);
			if (std::isinf(std::exp(lF - P.lFmax)))
				std::cerr << "Found inf result in integrand for amax = "
					      << amax << ", a = " << a << ".\n";
			return std::exp(lF - P.lFmax);
		};

		exp_sinh<double> es;
		const double I = es.integrate(integrand3, term, &err, &L1);
		condition_warn(I, err, L1);
		res += std::log(I);
	}

	/* Result: */
	return res;
}

#endif



double GammaConjugatePriorLogLikelihood::ln_Phi(double lp, double ls, double n,
                                                double v, double epsabs,
                                                double epsrel)
{
	/* Return: */
	return ln_Phi_backend(lp, ls, n, v, epsabs, epsrel);
}



/*
 * Kullback-Leibler distance:
 */
double GammaConjugatePriorLogLikelihood::kullback_leibler(
            double lp, double s, double n, double v,
            double lp_ref, double s_ref, double n_ref,
            double v_ref, double epsabs, double epsrel)
{
	const double ls = log(s);
	const double ls_ref = log(s_ref);
	const double ln_Phi = ln_Phi_backend(lp, ls, n, v, epsabs, epsrel);
	const double ln_Phi_ref = ln_Phi_backend(lp_ref, ls_ref, n_ref, v_ref,
	                                         epsabs, epsrel);

	/* Now the integral: */
	auto integrand = [=](double a) -> double {
		a += GCP_AMIN;
		return std::exp((a - 1.0)*lp - n*lgamma(a) + lgamma(a*v) - v * a * ls
		                - ln_Phi)
		       * ( (a - 1.0)*(lp - lp_ref) - (n - n_ref) * lgamma(a)
		            - a * v * (s - s_ref) / s
		            + a * (v - v_ref) * (digamma(a*v) - ls));
	};

	exp_sinh<double> es;
	constexpr double term = std::sqrt(std::numeric_limits<double>::epsilon());
	double err, L1;
	const double I = es.integrate(integrand, term, &err, &L1);
	condition_warn(I, err, L1);

	return ln_Phi_ref - ln_Phi + I;
}




/*****************************************************
 *
 *                 Typical code.
 *
 *****************************************************/


static std::vector<GammaConjugatePriorLogLikelihood::ab_t>
compute_ab(const double* a, const double* b, size_t N)
{
	std::vector<GammaConjugatePriorLogLikelihood::ab_t> ab(N);
	for (size_t i=0; i<N; ++i){
		ab[i].a = a[i];
	}
	for (size_t i=0; i<N; ++i){
		ab[i].b = b[i];
	}
	return ab;
}


GammaConjugatePriorLogLikelihood::GammaConjugatePriorLogLikelihood(
         double p, double s, double n, double v, const std::vector<ab_t>& ab,
         double nv_surplus_min, double vmin, double epsabs, double epsrel
       )
	: nv_surplus_min(nv_surplus_min), vmin(vmin), epsrel(epsrel),
	  epsabs(epsabs), lp_(std::log(p)), p_(p),
	  ls(std::log(s)), s_(s), v_(std::max(v, vmin)),
	  nv(std::max(n/v_, 1 + nv_surplus_min)), n_(std::max(n, nv*v_)),
	  ab(ab), W(ab.size())
{
	for (const ab_t& ab_ : ab){
		if (ab_.a < GCP_AMIN || ab_.a <= 0.0)
			throw std::domain_error("a out of bounds ]amin, inf[.");
		if (ab_.b <= 0.0)
			throw std::domain_error("b out of bounds ]0, inf[.");
	}
	init_constants();
	_ints = integrals();
	integrals_cache.put(parameters(), _ints);
}


GammaConjugatePriorLogLikelihood::GammaConjugatePriorLogLikelihood(
         double p, double s, double n, double v, const double* a,
         const double* b, size_t Nab, double nv_surplus_min, double vmin,
         double epsabs, double epsrel
       )
	: nv_surplus_min(nv_surplus_min), vmin(vmin), epsrel(epsrel),
	  epsabs(epsabs), lp_(std::log(p)), p_(p),
	  ls(std::log(s)), s_(s), v_(std::max(v, vmin)),
	  nv(std::max(n/v_, 1 + nv_surplus_min)), n_(std::max(n, nv*v_)),
	  ab(compute_ab(a, b, Nab)), W(Nab)
{
	for (const ab_t& ab_ : ab){
		if (ab_.a < GCP_AMIN || ab_.a <= 0.0)
			throw std::domain_error("a out of bounds ]amin, inf[.");
		if (ab_.b <= 0)
			throw std::domain_error("b out of bounds ]0, inf[.");
	}
	init_constants();
	_ints = integrals();
	integrals_cache.put(parameters(), _ints);
}


std::unique_ptr<GammaConjugatePriorLogLikelihood>
GammaConjugatePriorLogLikelihood::make_unique(double p, double s, double n,
                                      double v, const double* a,
                                      const double* b, size_t Nab,
                                      double nv_surplus_min, double vmin,
                                      double epsabs, double epsrel)
{
	typedef GammaConjugatePriorLogLikelihood GCPLL;
	return std::make_unique<GCPLL>(p, s, n, v, a, b, Nab, nv_surplus_min, vmin,
	                               epsabs, epsrel);
}


ColumnVector GammaConjugatePriorLogLikelihood::parameters() const
{
	return ColumnVector({lp_, ls, nv, v_});
}

ColumnVector GammaConjugatePriorLogLikelihood::lower_bound() const
{
	constexpr double inf = std::numeric_limits<double>::infinity();
	return ColumnVector({-inf, -inf, 1.0 + nv_surplus_min, vmin});
}

ColumnVector GammaConjugatePriorLogLikelihood::upper_bound() const
{
	return ColumnVector({std::numeric_limits<double>::infinity(),
	                     std::numeric_limits<double>::infinity(),
	                     std::numeric_limits<double>::infinity(),
	                     std::numeric_limits<double>::infinity()});
}


double GammaConjugatePriorLogLikelihood::operator()() const
{
	/* Unweighted likelihood: */
	const size_t N = ab.size();
	const double s0 = v_ * albsum - lbsum;
	const double s1 = lp_*asum - N*lp_;
	const double s2 = -s_*bsum;
	const double s3 = -n_*lgasum;
	return s0 + s1 + s2 + s3 - N*_ints.lPhi;
}


ColumnVector GammaConjugatePriorLogLikelihood::gradient() const
{
	/* Use numerical differentiation to derive log of normalization constant: */
	ColumnVector res;
	res[0] = -W * 0.5 * (_ints.forward[0] - _ints.backward[0]) / delta;
	res[1] = -W * 0.5 * (_ints.forward[1] - _ints.backward[1]) / delta;
	res[2] = -W * 0.5 * (_ints.forward[2] - _ints.backward[2]) / delta;
	res[3] = -W * 0.5 * (_ints.forward[3] - _ints.backward[3]) / delta;

	/* Derivatives of the remainder by the parameters: */
	res[0] += -W + asum;
	res[1] += -s_ * bsum;
	res[2] += -v_ * lgasum;
	res[3] += albsum - nv * lgasum;

	return res;
}


SquareMatrix GammaConjugatePriorLogLikelihood::hessian() const
{
	SquareMatrix m;
	/* Numerical derivatives of ln(Phi), logarithm of the normalization
	 * constant: */
	const double id2 = 1.0 / (delta * delta);
	m(0,0) = -W * (_ints.forward[0] + _ints.backward[0] - 2*_ints.lPhi) * id2;
	m(1,1) = -W * (_ints.forward[1] + _ints.backward[1] - 2*_ints.lPhi) * id2;
	m(2,2) = -W * (_ints.forward[2] + _ints.backward[2] - 2*_ints.lPhi) * id2;
	m(3,3) = -W * (_ints.forward[3] + _ints.backward[3] - 2*_ints.lPhi) * id2;

	/* Mixed derivatives: Compute simple difference of the numerical gradient
	 * and the value with double increased spacing:*/
	const double fb0 = _ints.forward[0] - _ints.backward[0];
	const double fb1 = _ints.forward[1] - _ints.backward[1];
	const double fb2 = _ints.forward[2] - _ints.backward[2];
	m(0,1) = -W * ((_ints.fw_lp_ls - _ints.forward[0]) - 0.5*fb1) * id2;
	m(0,2) = -W * ((_ints.fw_lp_nv - _ints.forward[2]) - 0.5*fb0) * id2;
	m(0,3) = -W * ((_ints.fw_lp_v  - _ints.forward[3]) - 0.5*fb0) * id2;
	m(1,2) = -W * ((_ints.fw_ls_nv - _ints.forward[2]) - 0.5*fb1) * id2;
	m(1,3) = -W * ((_ints.fw_ls_v  - _ints.forward[3]) - 0.5*fb1) * id2;
	m(2,3) = -W * ((_ints.fw_nv_v  - _ints.forward[3]) - 0.5*fb2) * id2;

	/* Finally the analytic cross term of nv and v and the ls term: */
	m(1,1) -= s_ * bsum;
	m(2,3) -= lgasum;

	// Symmetry:
	m(1,0) = m(0,1);
	m(2,0) = m(0,2);
	m(3,0) = m(0,3);
	m(2,1) = m(1,2);
	m(3,1) = m(1,3);
	m(3,2) = m(2,3);

	return m;
}


void GammaConjugatePriorLogLikelihood::optimize()
{
	newton_optimize<GammaConjugatePriorLogLikelihood>(*this,
	                                     /* Default values proven successful: */
	                                             1e-2, // g
	                                             2.0, // gstep_down
	                                             1.3, // gstep_up
	                                             5e-2, // gmax
	                                             40000, // nmax
	                                             0.5, // armijo
	                                             0.01);  // armijo_gradient
}

double GammaConjugatePriorLogLikelihood::p() const
{
	return p_;
}

double GammaConjugatePriorLogLikelihood::lp() const
{
	return lp_;
}

double GammaConjugatePriorLogLikelihood::s() const
{
	return s_;
}

double GammaConjugatePriorLogLikelihood::n() const
{
	return n_;
}

double GammaConjugatePriorLogLikelihood::v() const
{
	return v_;
}


size_t GammaConjugatePriorLogLikelihood::data_count() const
{
	return ab.size();
}




void GammaConjugatePriorLogLikelihood::update(const ColumnVector& param)
{
	lp_ = param(0);
	ls = param(1);
	nv = param(2);
	v_ = param(3);
	p_ = std::exp(lp_);
	s_ = std::exp(ls);
	n_ = nv * v_;

	/* Caching the numerical integrals: */
	auto it = integrals_cache.find(param);
	if (it != integrals_cache.end()){
		_ints = *it;
	} else {
		_ints = integrals();
		integrals_cache.put(param, _ints);
	}
}


GammaConjugatePriorLogLikelihood::integrals_t
GammaConjugatePriorLogLikelihood::integrals() const
{
	/* 1. The integrals: */
	integrals_t ints;

	try {
		/* 1.1 ln(Phi) */
		ints.lPhi = ln_Phi_backend(lp_, ls, n_, v_, epsabs, epsrel);

		/* 1.2 The stencil of ln(Phi): */
		ints.forward[0] = ln_Phi_backend(lp_+delta, ls, n_, v_, epsabs, epsrel);
		ints.forward[1] = ln_Phi_backend(lp_, ls+delta, n_, v_, epsabs, epsrel);
		ints.forward[2] = ln_Phi_backend(lp_, ls, (nv+delta)*v_, v_, epsabs,
		                                 epsrel);
		ints.forward[3] = ln_Phi_backend(lp_, ls, nv*(v_+delta), v_+delta,
		                                 epsabs, epsrel);


		ints.backward[0] = ln_Phi_backend(lp_-delta, ls, n_, v_, epsabs,
		                                  epsrel);
		ints.backward[1] = ln_Phi_backend(lp_, ls-delta, n_, v_, epsabs,
		                                  epsrel);
		ints.backward[2] = ln_Phi_backend(lp_, ls, (nv-delta)*v_, v_, epsabs,
		                                  epsrel);
		ints.backward[3] = ln_Phi_backend(lp_, ls, nv*(v_-delta), v_-delta,
		                                  epsabs, epsrel);

		if (use_hessian){
			ints.fw_lp_ls = ln_Phi_backend(lp_+delta, ls+delta, n_, v_, epsabs,
			                               epsrel);
			ints.fw_lp_nv = ln_Phi_backend(lp_+delta, ls, (nv+delta)*v_, v_,
			                               epsabs, epsrel);
			ints.fw_lp_v = ln_Phi_backend(lp_+delta, ls, nv*(v_+delta),
			                              v_+delta, epsabs, epsrel);
			ints.fw_ls_nv = ln_Phi_backend(lp_, ls+delta, (nv+delta)*v_, v_,
			                               epsabs, epsrel);
			ints.fw_ls_v = ln_Phi_backend(lp_, ls+delta, nv*(v_+delta),
			                              v_+delta, epsabs, epsrel);
			ints.fw_nv_v = ln_Phi_backend(lp_, ls, (nv+delta)*(v_+delta),
			                              v_+delta, epsabs, epsrel);
		} else {
			ints.fw_lp_ls = ints.fw_lp_nv = ints.fw_lp_v = ints.fw_ls_nv
			     = ints.fw_ls_v = ints.fw_nv_v = 0.0;
		}
	} catch (std::runtime_error& e) {
		std::string msg("Failed to compute integrals at parameters p=");
		msg.append(std::to_string(p_));
		msg.append(", s=");
		msg.append(std::to_string(s_));
		msg.append(", n=");
		msg.append(std::to_string(n_));
		msg.append(", v=");
		msg.append(std::to_string(v_));
		msg.append(".\nError message:'");
		msg.append(e.what());
		msg.append("'");
		throw std::runtime_error(msg);
	}

	return ints;
}

void GammaConjugatePriorLogLikelihood::init_constants()
{
	/* Initialize the constants depending only on a and b values: */
	albsum = 0;
	lbsum = 0;
	asum = 0;
	bsum = 0;
	lgasum = 0;
	for (const ab_t& x : ab){
		const double lb = std::log(x.b);
		asum += x.a;
		bsum += x.b;
		lbsum += lb;
		albsum += x.a * lb;
		lgasum += std::lgamma(x.a);
	}
}
