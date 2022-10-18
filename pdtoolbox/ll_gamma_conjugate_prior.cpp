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
#include <optimize.hpp>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl_workspace.hpp>
#include <gsl_rootfinder.hpp>
#include <gsl/gsl_errno.h>

#include <iostream>
#include <iomanip>


/* Namespae: */
using namespace pdtoolbox;
using std::abs, std::max, std::min, std::log, std::exp, std::sqrt;



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

static double F(double a, void* parameters)
{
	const params0_t& params = *static_cast<params0_t*>(parameters);
	double lF = ln_F(a, params.lp, params.ls, params.n, params.v);
	return std::exp(lF - params.lFmax);
}



/* Compute the maximum of the integrand. */
static double amax_f0(double a, double v, double n)
{
	if (a > 0.1)
		return v*gsl_sf_psi(v*a) - n*gsl_sf_psi(a);
	/* For small values of a, use recurrence formula: */
	return v*gsl_sf_psi(v*a + 1) - n*gsl_sf_psi(a+1) + (n-1)/a;
}


static double amax_f1(double a, double v, double n)
{
	if (a > 1e8){
		/* Stirling's formula is accurate. */
		const double ia = 1/a;
		return (v-n)*ia + 0.5*(1-n)*ia*ia;
	}
	return v*v*gsl_sf_psi_1(v*a) - n*gsl_sf_psi_1(a);
}


static double amax_f2(double a, double v, double n)
{
//	if (a > 1e8){
//		/* Stirling's formula is accurate. */
//		const double ia = 1/a;
//		return (v-n)*ia + 0.5*(1-n)*ia*ia;
//	}
	return v*v*v*gsl_sf_psi_n(2,v*a) - n*gsl_sf_psi_n(2,a);
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


/* Compute the transition point where the integrand is dampened by 1e-20
 * compared to its maximum: */
static double alr_root_backend(double a, void* parameters)
{
	const params0_t& params = *static_cast<params0_t*>(parameters);
	const double leps = std::log(params.epsrel);
	double lF = ln_F(a, params.lp, params.ls, params.n, params.v);
	return lF - params.lFmax - leps;
}

static double alr_root(double al, double ar, void* params, double epsabs)
{
	/* Root finding: */
	RootFSolver solver(gsl_root_fsolver_brent);
	gsl_root_fsolver* fsolve = solver.get();
	if (!fsolve){
		throw std::runtime_error("Could not construct gsl_root_fsolver.\n");
	}
	gsl_function rootfun;
	rootfun.params = params;
	rootfun.function = &alr_root_backend;
	gsl_root_fsolver_set(fsolve, &rootfun, al, ar);

	double a;
	int status;
	for (int i=0; i<100; ++i){
		gsl_root_fsolver_iterate(fsolve);
		a = gsl_root_fsolver_root(fsolve);
		al = gsl_root_fsolver_x_lower(fsolve);
		ar = gsl_root_fsolver_x_upper(fsolve);
		status = gsl_root_test_interval(al, ar, epsabs, 1e-3);
		if (status == GSL_SUCCESS)
			break;
	}
	if (status != GSL_SUCCESS){
		throw std::runtime_error("Could not determine root.");
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

static bounds_t
peak_integration_bounds(const double amax, const double epsrel,
                        const double epsabs, const double f2_amax, params0_t& P)
{
	const double nleps0 = -log(epsrel) + log(1e4);
	const double leps = log(epsrel);
	const double da = sqrt(2/f2_amax * nleps0);
	double ar = amax + da;
	while (ln_F(ar, P.lp, P.ls, P.n, P.v) - P.lFmax >= leps)
	   ar *= 1.5;

	double al = max(amax-da, 0.0);
	if (ln_F(0, P.lp, P.ls, P.n, P.v) - P.lFmax < leps){
		if (ln_F(al, P.lp, P.ls, P.n, P.v) - P.lFmax >= leps)
			al = alr_root(0, al, &P, epsabs);
		else
			al = alr_root(al, amax, &P, epsabs);
	}
	ar = alr_root(amax, ar, &P, epsabs);
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




/* Compute the natural logarithm of the integration constant: */
static double ln_Phi_backend(double lp, double ls, double n, double v,
                             double epsabs, double epsrel, size_t workspace_N,
                             gsl_integration_workspace* workspace)
{
	/* Get a paramter structure: */
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
	double S=0, Stmp, abserr;
	gsl_function integrand;
	integrand.params = &P;
	integrand.function = &F;
	double res = 0.0 ;
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
			double interval[2] = {0, bounds.l};
			gsl_integration_qagp(&integrand, interval, 2, epsabs, epsrel,
			                     workspace_N, workspace, &Stmp, &abserr);
			S += Stmp;
		}

		/* A shortcut to Laplace approximation for very large amax: */
		const double da_l = amax - bounds.l, da_r = bounds.r - amax;
		if (amax > 1e10){
			/* For consistency checking, make sure that the first non-quadratic
			 * term of the Taylor expansion of the log-integrand (the cubic
			 * term) is less than 4.61 (=log(1e2)) at the location of the
			 * cutoff: */
			//  f3 = v*v*v*gsl_sf_psi_n(2,v*amax) - n*gsl_sf_psi_n(2,amax);
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
		gsl_integration_qag(&integrand, bounds.l, bounds.r, epsabs, epsrel,
		                    workspace_N, GSL_INTEG_GAUSS51, workspace, &Stmp,
		                    &abserr);
		S += Stmp;
		res += std::log(S);
	} else {
		/* Integrate the singularity at a=0 and for the rest, integrate to
		 * infinity: */
		P.lFmax = ln_F(1.0, lp, ls, n, v);
		res = P.lFmax;
		double interval[2] = {0, 1.0};
		gsl_integration_qagp(&integrand, interval, 2, epsabs, epsrel,
		                     workspace_N, workspace, &Stmp, &abserr);
		S += Stmp;

		gsl_integration_qagiu(&integrand, 1.0, epsabs, epsrel, workspace_N,
		                      workspace, &Stmp, &abserr);
		S += Stmp;
		res += std::log(S);
	}

	/* Result: */
	return res;
}


double GammaConjugatePriorLogLikelihood::ln_Phi(double lp, double ls, double n,
                                                double v, double epsabs,
                                                double epsrel,
                                                size_t workspace_size)
{
	Workspace ws(workspace_size);

	// Set error handler:
	gsl_error_handler_t* error_handler \
	    = gsl_set_error_handler(&pdtoolbox::error_handler_cpp);

	/* Integrate: */
	const double res = ln_Phi_backend(lp, ls, n, v, epsabs, epsrel,
	                                  workspace_size, ws.get());

	if (error_handler)
		gsl_set_error_handler(error_handler);

	/* Return: */
	return res;
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
         const std::vector<double>& w, double nv_surplus_min, double vmin,
         double epsabs, double epsrel
       )
	: nv_surplus_min(nv_surplus_min), vmin(vmin), epsrel(epsrel),
	  epsabs(epsabs), lp(std::log(p)), p_(p),
	  ls(std::log(s)), s_(s), v_(std::max(v, vmin)),
	  nv(std::max(n/v_, 1 + nv_surplus_min)), n_(std::max(n, nv*v_)),
	  w(w), ab(ab),
	  W((w.size() > 0) ? std::accumulate(w.begin(), w.end(), 0) : ab.size())
{
	for (const ab_t& ab_ : ab){
		if (ab_.a <= 0)
			throw std::domain_error("a out of bounds ]0, inf[.");
		if (ab_.b <= 0)
			throw std::domain_error("b out of bounds ]0, inf[.");
	}
	init_constants();
	_ints = integrals();
	integrals_cache.put(parameters(), _ints);
}


GammaConjugatePriorLogLikelihood::GammaConjugatePriorLogLikelihood(
         double p, double s, double n, double v, const double* a,
         const double* b, size_t Nab, const double* w_, size_t Nw,
         double nv_surplus_min, double vmin, double epsabs, double epsrel
       )
	: nv_surplus_min(nv_surplus_min), vmin(vmin), epsrel(epsrel),
	  epsabs(epsabs), lp(std::log(p)), p_(p),
	  ls(std::log(s)), s_(s), v_(std::max(v, vmin)),
	  nv(std::max(n/v_, 1 + nv_surplus_min)), n_(std::max(n, nv*v_)),
	  w(w_, w_+Nw), ab(compute_ab(a, b, Nab)),
	  W((Nw > 0) ? std::accumulate(w.begin(), w.end(), 0) : Nab)
{
	for (const ab_t& ab_ : ab){
		if (ab_.a <= 0)
			throw std::domain_error("a out of bounds ]0, inf[.");
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
                                      const double* w_, size_t Nw,
                                      double nv_surplus_min, double vmin,
                                      double epsabs, double epsrel)
{
	return std::make_unique<GammaConjugatePriorLogLikelihood>(p, s, n, v, a,
	                                                      b, Nab, w_, Nw,
	                                                      nv_surplus_min, vmin,
	                                                      epsabs, epsrel);
}


Vector<D4> GammaConjugatePriorLogLikelihood::parameters() const
{
	return Vector<D4>({lp, ls, nv, v_});
}

Vector<D4> GammaConjugatePriorLogLikelihood::lower_bound() const
{
	constexpr double inf = std::numeric_limits<double>::infinity();
	return Vector<D4>({-inf, -inf, 1.0 + nv_surplus_min, vmin});
}

Vector<D4> GammaConjugatePriorLogLikelihood::upper_bound() const
{
	return Vector<D4>({std::numeric_limits<double>::infinity(),
	                   std::numeric_limits<double>::infinity(),
	                   std::numeric_limits<double>::infinity(),
	                   std::numeric_limits<double>::infinity()});
}


double GammaConjugatePriorLogLikelihood::operator()() const
{
	if (w.size() == 0){
		/* Unweighted likelihood: */
		const size_t N = ab.size();
		const double s0 = v_ * albsum - lbsum;
		const double s1 = lp*asum - N*lp;
		const double s2 = -s_*bsum;
		const double s3 = -n_*lgasum;
		return s0 + s1 + s2 + s3 - N*_ints.lPhi;
	} else {
		/* Weighted likelihood: */
		// TODO FIXME
		throw std::runtime_error("Weighted GCP log-likelihood not yet "
		                         "implemented.");
	}
}


Vector<D4> GammaConjugatePriorLogLikelihood::gradient() const
{
	/* Use numerical differentiation to derive log of normalization constant: */
	Vector<D4> res;
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


SquareMatrix<D4> GammaConjugatePriorLogLikelihood::hessian() const
{
	SquareMatrix<D4> m;
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


ColumnSpecifiedMatrix<D4> GammaConjugatePriorLogLikelihood::jacobian() const
{
	/* Not implemented yet. */
	throw std::runtime_error("TODO : Implement LogLogistic Jacobian!");
}


void GammaConjugatePriorLogLikelihood::optimize()
{
	newton_optimize<GammaConjugatePriorLogLikelihood,D4>(*this,
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




void GammaConjugatePriorLogLikelihood::update(const Vector<D4>& param)
{
	lp = param[0];
	ls = param[1];
	nv = param[2];
	v_ = param[3];
	p_ = std::exp(lp);
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
	constexpr unsigned int workspace_size = 2000;
	Workspace ws(workspace_size);
	gsl_error_handler_t* error_handler \
	    = gsl_set_error_handler(&pdtoolbox::error_handler_cpp);
	integrals_t ints;

	try {
		/* 1.1 ln(Phi) */
		ints.lPhi = ln_Phi_backend(lp, ls, n_, v_, epsabs, epsrel,
		                           workspace_size, ws.get());

		/* 1.2 The stencil of ln(Phi): */
		ints.forward[0] = ln_Phi_backend(lp+delta, ls, n_, v_, epsabs, epsrel,
		                                 workspace_size, ws.get());
		ints.forward[1] = ln_Phi_backend(lp, ls+delta, n_, v_, epsabs, epsrel,
		                                 workspace_size, ws.get());
		ints.forward[2] = ln_Phi_backend(lp, ls, (nv+delta)*v_, v_, epsabs,
		                                 epsrel, workspace_size, ws.get());
		ints.forward[3] = ln_Phi_backend(lp, ls, nv*(v_+delta), v_+delta,
		                                 epsabs, epsrel, workspace_size,
		                                 ws.get());


		ints.backward[0] = ln_Phi_backend(lp-delta, ls, n_, v_, epsabs, epsrel,
		                                  workspace_size, ws.get());
		ints.backward[1] = ln_Phi_backend(lp, ls-delta, n_, v_, epsabs, epsrel,
		                                  workspace_size, ws.get());
		ints.backward[2] = ln_Phi_backend(lp, ls, (nv-delta)*v_, v_, epsabs,
		                                  epsrel, workspace_size, ws.get());
		ints.backward[3] = ln_Phi_backend(lp, ls, nv*(v_-delta), v_-delta,
		                                  epsabs, epsrel, workspace_size,
		                                  ws.get());

		if (use_hessian){
			ints.fw_lp_ls = ln_Phi_backend(lp+delta, ls+delta, n_, v_, epsabs,
			                               epsrel, workspace_size, ws.get());
			ints.fw_lp_nv = ln_Phi_backend(lp+delta, ls, (nv+delta)*v_, v_,
			                               epsabs, epsrel, workspace_size,
			                               ws.get());
			ints.fw_lp_v = ln_Phi_backend(lp+delta, ls, nv*(v_+delta), v_+delta,
			                              epsabs, epsrel, workspace_size,
			                              ws.get());
			ints.fw_ls_nv = ln_Phi_backend(lp, ls+delta, (nv+delta)*v_, v_,
			                               epsabs, epsrel, workspace_size,
			                               ws.get());
			ints.fw_ls_v = ln_Phi_backend(lp, ls+delta, nv*(v_+delta), v_+delta,
			                              epsabs, epsrel, workspace_size,
			                              ws.get());
			ints.fw_nv_v = ln_Phi_backend(lp, ls, (nv+delta)*(v_+delta),
			                              v_+delta, epsabs, epsrel,
			                              workspace_size, ws.get());
		} else {
			ints.fw_lp_ls = ints.fw_lp_nv = ints.fw_lp_v = ints.fw_ls_nv
			     = ints.fw_ls_v = ints.fw_nv_v = 0.0;
		}
	} catch (std::runtime_error& e) {

		if (error_handler)
			gsl_set_error_handler(error_handler);
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

	if (error_handler)
		gsl_set_error_handler(error_handler);

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
