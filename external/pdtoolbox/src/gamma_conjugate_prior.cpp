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
#include <cmath>
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
	double a = max(exp((P.lp - P.v*P.ls + P.v*log(P.v)) / (P.n-P.v)),
	               max(1.0, 1.1*amin));

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
		a = max(a, amin);
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
		const double a1 = max(a0 - f0/f1, 1e-12);
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
//static double add_logs(double la, double lb)
//{
//	constexpr double leps = log(1e-16);
//	double lmin = min(la,lb);
//	double lmax = max(la,lb);
//	if (lmin < lmax + leps){
//		return lmax;
//	}
//	return lmax + log1p(exp(lmin-lmax));
//}


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

static double ln_Phi_backend(double lp, double ls, double n, double v,
                             double amin, double epsabs, double epsrel)
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
	constexpr double term = cnst_sqrt(std::numeric_limits<double>::epsilon());

	auto integrand1 = [&](double a, double distance_to_next_bound) -> double {
		double lF = ln_F(a, P.lp, P.ls, P.n, P.v);
		if (isinf(exp(lF - P.lFmax)))
			std::cerr << "Found inf result in integrand for amax = "
			          << amax << ", a = " << a << ".\n";
		return exp(lF - P.lFmax);
	};

	if (n_extrema >= 1 && amax > amin){
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
			const double I = ts.integrate(integrand1, amin, amax, term,
			                              &err, &L1);
			S += I;
			condition_warn(I, err, L1, __LINE__, 3.0);
		}

		/* (2) From amax to inf: */
		{
			auto integrand2 = [&](double a) -> double {
				res = exp(ln_F(a+amax, P.lp, P.ls, P.n, P.v)
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
			exp_sinh<double> es;
			const double I = es.integrate(integrand2, term, &err, &L1);
			S += I;
			condition_warn(I, err, L1, __LINE__, 3.0);
		}
		return P.lFmax + log(S);
	} else {
		/* Integrate the singularity at a=0 and for the rest, integrate to
		 * infinity: */
		P.lFmax = ln_F(max(amin, 1.0), lp, ls, n, v);
		res = P.lFmax;

		auto integrand3 = [&](double a) -> double {
			double lF = ln_F(a+amin, P.lp, P.ls, P.n, P.v);
			if (isinf(exp(lF - P.lFmax)))
				std::cerr << "Found inf result in integrand for amax = "
					      << amax << ", a = " << a << ".\n";
			return exp(lF - P.lFmax);
		};

		exp_sinh<double> es;
		const double I = es.integrate(integrand3, term, &err, &L1);
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
double GammaConjugatePriorBase::kullback_leibler(
            double lp, double s, double n, double v,
            double lp_ref, double s_ref, double n_ref,
            double v_ref, double amin, double epsabs, double epsrel)
{
	const double ls = log(s);
	const double ls_ref = log(s_ref);
	const double ln_Phi = ln_Phi_backend(lp, ls, n, v, amin, epsabs, epsrel);
	const double ln_Phi_ref = ln_Phi_backend(lp_ref, ls_ref, n_ref, v_ref,
	                                         amin, epsabs, epsrel);

	/* Now the integral: */
	auto integrand = [=](double a) -> double {
		a += amin;
		const double lga = lgamma(a);
		const double C1 = (lp - lp_ref) - v * (s - s_ref) / s
		                  - ls * (v - v_ref);
		const double C0 = (lp_ref - lp);
		const double C2 = - (n - n_ref);
		const double C3 =  (v - v_ref);
		return exp((a - 1.0)*lp - n*lga + lgamma(a*v) - v * a * ls
		           - ln_Phi)
		       * ((C1 + C3 * digamma(a*v)) * a  +  C0  +  C2 * lga);
	};

	exp_sinh<double> es;
	constexpr double term = cnst_sqrt(std::numeric_limits<double>::epsilon());
	double err, L1;
	const double I = es.integrate(integrand, term, &err, &L1);

	/* The integrand changes sign due to non-exponentiated part.
	 * A moderately large condition number (~1e3) occurs rather frequently
	 * without the integrand being highly oscillatory (typically
	 * perhaps one change of sign).
	 * Warn for a condition number of 1e5 - this should still give
	 * sufficient precision from the remaining ~10 digits.
	 */
	condition_warn(I, err, L1, __LINE__, 1e5);

	return ln_Phi_ref - ln_Phi + I;
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
