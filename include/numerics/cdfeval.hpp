/*
 * A class to evaluate cumulative distribution functions from unnormed
 * densities.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Malte J. Ziebarth
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

#ifndef REHEATFUNQ_NUMERICS_CDFEVAL_HPP
#define REHEATFUNQ_NUMERICS_CDFEVAL_HPP

#include <vector>
#include <iostream>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>


namespace reheatfunq {

template<typename real>
class CDFEval {
public:
	template<typename func>
	CDFEval(const std::vector<real>& x, func F,
	        const real xmin, const real xmax,
	        const real outside_mass = 0.0,
	        const real dest_tol = 1e-8);

	real cdf(const real& x) const;

	real tail(const real& x) const;

	void add_outside_mass(const real mass, bool above=true);

	real norm() const {
		return _norm;
	};

private:
	struct xy_t {
		real x;
		real cdf;
		real tail;

		bool operator<(const xy_t& other) {
			return x < other.x;
		};
	};
	std::vector<xy_t> intervals;

	real _norm;
	real mass_above;
};


template<typename real>
template<typename func>
CDFEval<real>::CDFEval(const std::vector<real>& x, func F, const real xmin,
                       const real xmax, const real outside_mass,
                       const real dest_tol)
   : intervals(std::max<size_t>(x.size(),1)+1), _norm(0.0)
{
	/* Typedefs: */
	using boost::multiprecision::abs;
	using std::abs;
	using boost::math::quadrature::gauss_kronrod;
	typedef gauss_kronrod<real, 7> GK;

	const real eps_tol
	   = boost::math::tools::root_epsilon<double>();

	/* Setup all the intervals: */
	const size_t N = x.size();
	intervals[0].x = xmin;
	intervals[0].cdf = 0.0;
	if (N > 0){
		for (size_t i=0; i<N; ++i){
			intervals[i+1].x = x[i];
		}
	}
	std::sort(intervals.begin(), intervals.end());
	if (intervals.back().x > xmax || intervals.front().x < xmin)
		throw std::runtime_error("One of the x values is out of bounds.");

	/* Make sure that the intervals are finite: */
	auto dest = intervals.begin();
	for (auto from = intervals.cbegin(); from != intervals.cend(); ++from){
		if (from->x == dest->x)
			continue;
		++dest;
		*dest = *from;
	}
	intervals.resize((dest - intervals.begin()) + 1);

	/* Integrate all the sub intervals: */
	struct delta_error_t {
		real y;
		real yerr;
	};
	std::vector<delta_error_t> delta_err(intervals.size());
	auto it_err = delta_err.begin();
	typename std::vector<xy_t>::iterator it_next;
	real cdf = 0.0;
	for (auto it = intervals.begin(); it != intervals.end(); it = it_next){
		/* Find the right boundary of the integration interval: */
		it_next = it+1;
		real xr = (it_next == intervals.end()) ? xmax : it_next->x;

		/* Perform a one-shot Gauss-Kronrod quadrature: */
		it_err->y = GK::integrate(F, it->x, xr, 0, eps_tol, &it_err->yerr);

		/* Compute the cumulative sum: */
		cdf += it_err->y;
		if (it_next != intervals.end())
			it_next->cdf = cdf;

		++it_err;
	}

	/* Tail distribution: */
	real tail = 0.0;
	auto rit_err = delta_err.rbegin();
	for (auto rit = intervals.rbegin(); rit != intervals.rend(); ++rit){
		/* Since we start from the beyond-the-end xmax, the last
		 * element is at the beginning of the interval and has
		 * a tail distribution value equal to its integral.
		 *
		 * Each element gets assigned the current value of the tail
		 * distribution.
		 */
		tail += rit_err->y;
		rit->tail = tail;

		++rit_err;
	}

	/* After the one-shot integration, investigate the errors. */
	it_err = delta_err.begin();
	cdf = 0.0;
	real err = 0.0;
	size_t refinements = 0;
	for (auto it = intervals.begin(); it != intervals.end(); it = it_next){
		it_next = it+1;

		/* Check if we want to refine this integral: */
		real compare = it->tail;
		if (it_next != intervals.end())
			compare = std::max(it_next->cdf, compare);

		if (it_err->yerr > compare * dest_tol){
			++refinements;

			/* Find the right boundary of the integration interval: */
			real xr = (it_next == intervals.end()) ? xmax : it_next->x;

			real local_tol
			   = std::max<real>(1e-2 * dest_tol * it->cdf / it_err->y,
			                    eps_tol);
			it_err->y = GK::integrate(F, it->x, xr, 8, local_tol,
			                          &it_err->yerr);
		}

		/* Compute the cumulative sum: */
		cdf += it_err->y;
		if (it_next != intervals.end())
			it_next->cdf = cdf;

		/* Cumulative error: */
		err += it_err->yerr;

		++it_err;
	}

	/* Tail distribution, see above: */
	tail = 0.0;
	rit_err = delta_err.rbegin();
	for (auto rit = intervals.rbegin(); rit != intervals.rend(); ++rit){
		tail += rit_err->y;
		rit->tail = tail;

		++rit_err;
	}

	/* Norm: */
	real scale = (outside_mass > 0) ? 1.0 / (cdf * (1.0 + outside_mass))
	                                : 1.0 / cdf;
	for (xy_t& xy : intervals){
		xy.cdf *= scale;
		xy.tail *= scale;
		xy.tail += outside_mass;
	}
	_norm = cdf;

	/* Error estimate (should be in the order of precision µ): */
#ifdef DEBUG
	real error = 0.0;
	for (auto it = intervals.begin(); it != intervals.end(); ++it){
		error = std::max(error, abs(it->tail - (1.0 - it->cdf)));
	}
	std::cout << "1 - (tail + cdf) error: " << error << "\n";
#endif
};


template<typename real>
real CDFEval<real>::cdf(const real& x) const
{
	/* Find the first element that is not smaller than x: */
	auto it = std::lower_bound(intervals.cbegin(), intervals.cend(), x,
	                           [](const xy_t& ele, const real& val) -> bool {
	                               return ele.x < val;
	                           }
	);

	/* Shortcut for out of bounds: */
	if (it == intervals.cend()){
		return 1.0;
	}

	/* Check if it's the desired value: */
	if (it->x == x){
		return it->cdf;
	}

	/* Shortcut for out of bounds: */
	if (it == intervals.cbegin()){
		return 0.0;
	}

	std::cout << std::setprecision(30);
	std::cout << "before= " << (it-1)->x << "\n";
	std::cout << "x =     " << x << "\n";
	std::cout << "it->x = " << it->x << "\n";


	throw std::runtime_error("CDFEval::cdf for non-prior-provided nodes not "
	                         "implemented.");
}


template<typename real>
real CDFEval<real>::tail(const real& x) const
{
	/* Find the first element that is not smaller than x: */
	auto it = std::lower_bound(intervals.cbegin(), intervals.cend(), x,
	                           [](const xy_t& ele, const real& val) -> bool {
	                               return ele.x < val;
	                           }
	);

	/* Shortcut for out of bounds: */
	if (it == intervals.cend()){
		return mass_above;
	}

	/* Check if it's the desired value: */
	if (it->x == x){
		return it->tail;
	}

	/* Shortcut for out of bounds: */
	if (it == intervals.cbegin()){
		return 1.0;
	}

	std::cout << std::setprecision(30);
	std::cout << "before= " << (it-1)->x << "\n";
	std::cout << "x =     " << x << "\n";
	std::cout << "it->x = " << it->x << "\n";


	throw std::runtime_error("CDFEval::tail for non-prior-provided nodes not "
	                         "implemented.");
}


template<typename real>
void CDFEval<real>::add_outside_mass(const real mass, bool above)
{
	if (above){
		const real scale = 1.0 / (1.0 + mass);
		for (xy_t& xy : intervals){
			xy.cdf *= scale;
			xy.tail *= scale;
			xy.tail += mass;
		}
		mass_above = mass;
	} else
		throw std::runtime_error("not implemented.");

#ifdef DEBUG
	real error = 0.0;
	for (auto it = intervals.begin(); it != intervals.end(); ++it){
		error = std::max(error, abs(it->tail - (1.0 - it->cdf)));
	}
	std::cout << "1 - (tail + cdf) error: " << error << "\n";
#endif
}

}

#endif