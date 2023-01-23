/*
 * Quantile inversion using an interval halving tree with Gauss-Kronrod
 * quadrature. This file covers the tree leaves.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ
 *               2023 Malte J. Ziebarth
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

#include <functional>
#include <array>
#include <memory>
#include <iostream>

#ifndef PDTOOLBOX_QUANTILETREE_LEAF_HPP
#define PDTOOLBOX_QUANTILETREE_LEAF_HPP

namespace pdtoolbox {

/*
 * Forward declare the quantile tree that will reference the leaves:
 */
template<typename real>
class QuantileTreeBranch;


template<typename real>
class QuantileTreeLeaf {
	friend QuantileTreeBranch<real>;
public:
	QuantileTreeLeaf(std::function<real(real)> pdf, real xmin,
	                 real xmax, bool parallel=false);

	QuantileTreeLeaf(const QuantileTreeLeaf& other) = default;
	QuantileTreeLeaf(QuantileTreeLeaf&& other) = default;

	/* Mostly for internal use: */
	struct xf_t {
		real x;
		real f;
	};
	QuantileTreeLeaf(std::function<real(real)> pdf, xf_t left, xf_t right,
	                 bool parallel=false);

	/* Retrieving the integrals approximations of the G7, K15
	 * scheme: */
	real gauss() const;
	real kronrod() const;

	/* Boundaries: */
	real xmin() const;
	real xmax() const;


	/* Splitting the interval: */
	std::array<std::unique_ptr<QuantileTreeLeaf>,3>
	   split(std::function<real(real)> pdf, real z0, real z1) const;

	/* Estimating quantiles using trapezoidal rule (*local* quantiles
	 * within this integral): */
	template<uint_fast8_t n>
	std::array<real,n> trapezoidal_quantiles(std::array<real,n>&& y) const;


private:
	/* Function evaluations of the Gauss-Kronrod 7-15
	 * (plus the function evaluations at the end): */
	std::array<real,17> gk75_lr;
	real _xmin;
	real _xmax;
	real _kronrod;

	/* We don't typically need to cache the evaluation of the
	 * Gauss-Legendre quadrature since it is used only once
	 * for error estimation:
	 */
	#ifdef PDTOOLBOX_QUANTILEINVERTER_CACHE_GAUSS
	real _gauss;
	#endif

	real compute_gauss() const;
	real compute_kronrod() const;

	/*
	 * Static arrays:
	 */
	const static std::array<real,8> gk75_abscissa;
	const static std::array<real,8> k7_weights;
	const static std::array<real,4> g5_weights;
	const static std::array<real,16> k7_trapezoid_mass;
	const static std::array<real,17> scaled_k7_lr_abscissa;

	static std::array<real,16> compute_trapezoid_mass();
	static std::array<real,17> compute_scaled_k7_lr_abscissa();
};




template<typename real>
template<uint_fast8_t n>
std::array<real,n>
QuantileTreeLeaf<real>::trapezoidal_quantiles(std::array<real,n>&& y) const
{
	/*
	 * Note: This returns quantiles within the range [0,1], counted
	 *       relative to [xmin,xmax].
	 */

	/* Setup the trapezoidal quantiles: */
	std::array<real,16> trapez;
	real S = 0.0;
	for (uint_fast8_t i=0; i<16; ++i){
		trapez[i] = 0.5 * k7_trapezoid_mass[i] * (gk75_lr[i] + gk75_lr[i+1]);
		S += trapez[i];
	}

	/* Normalize to local quantile: */
	real scale = 1.0 / S;
	for (uint_fast8_t i=0; i<16; ++i){
		trapez[i] *= scale;
	}

	/* Now find the quantiles: */
	std::array<real, n> result;
	for (uint_fast8_t i=0; i<n; ++i){
		/* Find the quantile: */
		real last_q = 0.0;
		for (uint_fast8_t j=0; j<16; ++j){
			real next_q = last_q + trapez[j];
			if (next_q >= y[i]){
				/* Linearly interpolate the x boundaries of the interval
				 * [j,j+1]: */
				result[i] = (  (next_q - y[i]) * scaled_k7_lr_abscissa[j]
				             + (y[i] - last_q) * scaled_k7_lr_abscissa[j+1]
				            ) / trapez[j];

				/* Depending on whether the integrand is nearly constant
				 * or has a significant slope, compute the inverted quantile
				 * either linearly or by solving the quadratic integrated
				 * slope.
				 *
				 * Note: This following implementation has some algebraic
				 *       error in it.
				 */
				//double df = (gk75_lr[j+1] - gk75_lr[j]);
				//if (std::abs(df) >= 1e-8 * k7_trapezoid_mass[j]){
				//	/* Solving the quadratic function, i.e. the integral of
				//	 * the trapezoid: */
				//	double fb_fa_b_a = df / k7_trapezoid_mass[j];
				//	double g0 = 0.5 * fb_fa_b_a;
				//	double g1 = 0.5 * gk75_lr[j] / g0;
				//	double dy = y[i] - last_q;
				//	double dx = std::sqrt(g1*g1 + dy/(scale*g0)) - g1;
				//	double x0 = scaled_k7_lr_abscissa[j];
				//	if (dx < 0)
				//		throw std::runtime_error("dx < 0");
				//	else if (dx > scaled_k7_lr_abscissa[j+1]
				//	              - scaled_k7_lr_abscissa[j]){
				//	    std::cout << "dy = " << dy << "\n";
				//	    std::cout << "dx_max = " << scaled_k7_lr_abscissa[j+1]
				//	              - scaled_k7_lr_abscissa[j] << "\n";
				//	    std::cout << "dx = " << dx << "\n";
				//	    std::cout << "mass: " << k7_trapezoid_mass[j] << "\n";
				//		throw std::runtime_error("dx > interval.");
				//	}
				//	result[i] = x0 + dx;
				//}
				break;
			}
			last_q = next_q;
			if (j == 15){
				std::cerr << "y[i] = " << y[i] << "\n"
				          << "next_q=" << next_q << "\n";
				throw std::runtime_error("failed to set result[i].");
			}
		}
	}

	return result;
}


template<typename real>
QuantileTreeLeaf<real>::QuantileTreeLeaf(std::function<real(real)> pdf,
                                         real xmin, real xmax, bool parallel)
{
	/* Bounds: */
	_xmin = xmin;
	_xmax = xmax;

	/* Determine the coordinates of all points: */
	real xc = 0.5*(xmin+xmax);
	real dx = xmax - xmin;
	std::array<real,17> xi;

	xi[0] = xmin;
	for (uint_fast8_t i=1; i<8; ++i) {
		xi[i] = xc - 0.5 * gk75_abscissa[8-i] * dx;
	}
	xi[8] = xc;
	for (uint_fast8_t i=1; i<8; ++i) {
		xi[8+i] = xc + 0.5 * gk75_abscissa[i] * dx;
	}
	xi[16] = xmax;


	/* Evaluate at all the points */
	for (uint_fast8_t i=0; i<17; ++i){
		gk75_lr[i] = pdf(xi[i]);
	}


	/* Compute Gauss and Kronrod integrals: */
	_kronrod = compute_kronrod();

	#ifdef PDTOOLBOX_QUANTILEINVERTER_CACHE_GAUSS
	_gauss = compute_gauss();
	#endif
}


template<typename real>
QuantileTreeLeaf<real>::QuantileTreeLeaf(std::function<real(real)> pdf,
                                         QuantileTreeLeaf::xf_t left,
                                         QuantileTreeLeaf::xf_t right,
                                         bool parallel)
{
	/* Bounds: */
	_xmin = left.x;
	_xmax = right.x;

	/* Determine the coordinates of all points: */
	real xc = 0.5*(_xmin + _xmax);
	real dx = _xmax - _xmin;
	std::array<real,17> xi;

	xi[0] = _xmin;
	for (uint_fast8_t i=1; i<8; ++i) {
		xi[i] = xc - 0.5 * gk75_abscissa[8-i] * dx;
	}
	xi[8] = xc;
	for (uint_fast8_t i=1; i<8; ++i) {
		xi[8+i] = xc + 0.5 * gk75_abscissa[i] * dx;
	}
	xi[16] = _xmax;

	/* Evaluate at all the points
	 * (this restructuring allows for parallel evaluation of the
	 * integral loop - however, does not seem to benefit much,
	 * hence commented out currently). */
	gk75_lr[0] = left.f;
//	#pragma omp parallel for if(parallel)
	for (uint_fast8_t i=1; i<16; ++i){
		gk75_lr[i] = pdf(xi[i]);
	}
	gk75_lr[16] = right.f;


	/* Compute Gauss-Kronrod integral: */
	_kronrod = compute_kronrod();

	#ifdef PDTOOLBOX_QUANTILEINVERTER_CACHE_GAUSS
	_gauss = compute_gauss();
	#endif
}


template<typename real>
std::array<std::unique_ptr<QuantileTreeLeaf<real>>,3>
QuantileTreeLeaf<real>::split(std::function<real(real)> pdf, real z0,
                              real z1) const
{
	/* The x of splitting: */
	real x0 = (1.0 - z0) * _xmin + z0 * _xmax;
	real x1 = (1.0 - z1) * _xmin + z1 * _xmax;
	xf_t l({_xmin, gk75_lr[0]});
	xf_t c0({x0, pdf(x0)});
	xf_t c1({x1, pdf(x1)});
	xf_t r({_xmax, gk75_lr[16]});

	/* Return the array: */
	return std::array<std::unique_ptr<QuantileTreeLeaf<real>>,3>(
	          {std::make_unique<QuantileTreeLeaf<real>>(pdf, l, c0),
	           std::make_unique<QuantileTreeLeaf<real>>(pdf, c0, c1),
	           std::make_unique<QuantileTreeLeaf<real>>(pdf, c1, r)});
}


template<typename real>
real QuantileTreeLeaf<real>::gauss() const
{
	#ifdef PDTOOLBOX_QUANTILEINVERTER_CACHE_GAUSS
	return _gauss;
	#else
	return compute_gauss();
	#endif
}


template<typename real>
real QuantileTreeLeaf<real>::kronrod() const
{
	return _kronrod;
}


template<typename real>
real QuantileTreeLeaf<real>::xmin() const
{
	return _xmin;
}


template<typename real>
real QuantileTreeLeaf<real>::xmax() const
{
	return _xmax;
}


template<typename real>
real QuantileTreeLeaf<real>::compute_kronrod() const
{
	/* Gauss-Kronrod quadrature: */
	real I = 0.0;
	for (uint_fast8_t i=1; i<8; ++i){
		I += k7_weights[8-i] * gk75_lr[i];
	}
	I += k7_weights[0] * gk75_lr[8];
	for (uint_fast8_t i=9; i<16; ++i){
		I += k7_weights[i-8] * gk75_lr[i];
	}

	/* The coordinate transform
	 * Gauss-Kronrod quadrature formula integrates on the interval
	 * [-1,1], length 2, which we want to transform to [xmin,xmax]
	 */
	const real cscale = (_xmax - _xmin) / 2.0;

	return I * cscale;
}


template<typename real>
real QuantileTreeLeaf<real>::compute_gauss() const
{
	/* Gauss-Legendre quadrature: */
	real I = 0.0;
	for (uint_fast8_t i=1; i<4; ++i){
		I += g5_weights[4-i] * gk75_lr[2*i];
	}
	I += g5_weights[0] * gk75_lr[8];
	for (uint_fast8_t i=1; i<4; ++i){
		I += g5_weights[i] * gk75_lr[8 + 2*i];
	}

	/* The coordinate transform
	 * Gauss-Legendre quadrature formula integrates on the interval
	 * [-1,1], length 2, which we want to transform to [xmin,xmax]
	 */
	const real cscale = (_xmax - _xmin) / 2.0;

	return I * cscale;
}


template<typename real>
std::array<real,16> QuantileTreeLeaf<real>::compute_trapezoid_mass()
{
	/* Computes a mass matrix that sums up to 1 and
	 * in which mass[i] denotes the relative length of
	 * the interval between abscissa i-7 and abscissa i-6
	 * (where mass[6] is the interval left of 0 and mass[7]
	 *  the one right of 0). */
	std::array<real,16> mass;
	mass[0] = 0.5 * (1.0 - gk75_abscissa[7]);
	for (uint_fast8_t i=0; i<7; ++i){
		mass[i+1] = 0.5*(gk75_abscissa[7-i] - gk75_abscissa[6-i]);
	}
	for (uint_fast8_t i=7; i<14; ++i){
		mass[i+1] = 0.5*(gk75_abscissa[i-6] - gk75_abscissa[i-7]);
	}
	mass[15] = 0.5 * (1.0 - gk75_abscissa[7]);
	return mass;
}

/*
 * Points of the Kronrod abscissa transformed from [-1,1] to [0,1]
 */
template<typename real>
std::array<real,17> QuantileTreeLeaf<real>::compute_scaled_k7_lr_abscissa()
{
	std::array<real,17> scab;
	scab[0] = 0.0;
	for (uint_fast8_t i=0; i<7; ++i){
		scab[i+1] = 0.5 * (1.0 - gk75_abscissa[7-i]);
	}
	scab[8] = 0.5;
	for (uint_fast8_t i=8; i<15; ++i){
		scab[i+1] = 0.5 + 0.5 * gk75_abscissa[i-7];
	}
	scab[16] = 1.0;

	return scab;
}

} // namespace pdtoolbox

#endif