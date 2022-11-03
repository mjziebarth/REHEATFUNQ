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

#include <iostream>
#include <iomanip>
#include <string>


/* Namespae: */
using namespace pdtoolbox;
using std::abs, std::max, std::min, std::log, std::exp, std::sqrt;



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
         double nv_surplus_min, double vmin, double amin, double epsabs,
         double epsrel
       )
	: GammaConjugatePriorBase(std::log(p), s,
	                          std::max(n, (1.0 + nv_surplus_min)
	                                       *  (std::max(v, vmin))),
	                          std::max(v, vmin), amin, epsabs, epsrel),
	  nv_surplus_min(nv_surplus_min), vmin(vmin),
	  nv(std::max(n/v_, 1.0 + nv_surplus_min)),
	  ab(ab), W(ab.size())
{
	for (const ab_t& ab_ : ab){
		if (ab_.a < amin || ab_.a <= 0.0)
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
         double amin, double epsabs, double epsrel
       )
	: GammaConjugatePriorBase(std::log(p), s,
	                          std::max(n, (1.0 + nv_surplus_min)
	                                       *  (std::max(v, vmin))),
	                          std::max(v, vmin), amin, epsabs, epsrel),
	  nv_surplus_min(nv_surplus_min), vmin(vmin),
	  nv(std::max(n/v_, 1.0 + nv_surplus_min)),
	  ab(compute_ab(a, b, Nab)), W(Nab)
{
	for (const ab_t& ab_ : ab){
		if (ab_.a < amin || ab_.a <= 0.0)
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
                                      double amin, double epsabs, double epsrel)
{
	typedef GammaConjugatePriorLogLikelihood GCPLL;
	return std::make_unique<GCPLL>(p, s, n, v, a, b, Nab, nv_surplus_min, vmin,
	                               amin, epsabs, epsrel);
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
		ints.lPhi = ln_Phi(lp_, ls, n_, v_, amin, epsabs, epsrel);

		/* 1.2 The stencil of ln(Phi): */
		ints.forward[0] = ln_Phi(lp_+delta, ls, n_, v_, amin, epsabs, epsrel);
		ints.forward[1] = ln_Phi(lp_, ls+delta, n_, v_, amin, epsabs, epsrel);
		ints.forward[2] = ln_Phi(lp_, ls, (nv+delta)*v_, v_, amin, epsabs,
		                         epsrel);
		ints.forward[3] = ln_Phi(lp_, ls, nv*(v_+delta), v_+delta, amin, epsabs,
		                         epsrel);


		ints.backward[0] = ln_Phi(lp_-delta, ls, n_, v_, amin, epsabs, epsrel);
		ints.backward[1] = ln_Phi(lp_, ls-delta, n_, v_, amin, epsabs, epsrel);
		ints.backward[2] = ln_Phi(lp_, ls, (nv-delta)*v_, v_, amin, epsabs,
		                          epsrel);
		ints.backward[3] = ln_Phi(lp_, ls, nv*(v_-delta), v_-delta, amin,
		                          epsabs, epsrel);

		if (use_hessian){
			ints.fw_lp_ls = ln_Phi(lp_+delta, ls+delta, n_, v_, amin, epsabs,
			                       epsrel);
			ints.fw_lp_nv = ln_Phi(lp_+delta, ls, (nv+delta)*v_, v_, amin,
			                       epsabs, epsrel);
			ints.fw_lp_v = ln_Phi(lp_+delta, ls, nv*(v_+delta), v_+delta, amin,
			                      epsabs, epsrel);
			ints.fw_ls_nv = ln_Phi(lp_, ls+delta, (nv+delta)*v_, v_, amin,
			                       epsabs, epsrel);
			ints.fw_ls_v = ln_Phi(lp_, ls+delta, nv*(v_+delta), v_+delta, amin,
			                      epsabs, epsrel);
			ints.fw_nv_v = ln_Phi(lp_, ls, (nv+delta)*(v_+delta), v_+delta,
			                      amin, epsabs, epsrel);
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
