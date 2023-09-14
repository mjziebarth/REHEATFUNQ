/*
 * Barycentric Lagrange Interpolator.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022-2023 Malte J. Ziebarth
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

#ifndef REHEATFUNQ_NUMERICS_BARYLAGRINT_HPP
#define REHEATFUNQ_NUMERICS_BARYLAGRINT_HPP

#include <vector>
#include <cmath>
#include <stack>
#include <limits>
#include <utility>
#include <variant>
#include <iostream>
#include <stdexcept>

#include <boost/math/tools/minima.hpp>

namespace reheatfunq {
namespace numerics {

/*
 * Types for summation within the BarycentricLagrangeInterpolator
 * so as not to lose precision:
 */
template<typename real>
struct BLI_summand {
};

template<>
struct BLI_summand<double>{
	typedef long double type;
};

template<>
struct BLI_summand<float>{
	typedef double type;
};

//TODO Add Kahan-sum types here for long double and above:


/*
 * The main class:
 */

template<typename real>
class BarycentricLagrangeInterpolator
{
public:
	template<typename fun_t>
	BarycentricLagrangeInterpolator(fun_t func, real xmin, real xmax,
	            real tol_rel = std::sqrt(std::numeric_limits<real>::epsilon()),
	            real tol_abs = std::numeric_limits<real>::infinity(),
	            real fmin = -std::numeric_limits<real>::infinity(),
	            real fmax = std::numeric_limits<real>::infinity(),
	            size_t max_splits = 10)
	   : xmin(xmin), fmin(fmin), fmax(fmax),
	     ranges(determine_ranges(func, xmin, xmax, tol_rel, tol_abs, fmin,
	                             fmax, max_splits))
	{
		/* This check should never trigger, but let's keep it here to ensure
		 * safety of assuming that ranges is not empty.
		 */
		if (ranges.empty())
			throw std::runtime_error("Ended up with an empty sample range in "
			                         "BarycentricLagrangeInterpolator.");
	}


	real operator()(real x) const {
		if (x < xmin || x > ranges.back().xmax)
			throw std::runtime_error("x out of bounds in BarycentricLagrange"
			                         "Interpolator.");
		/* Find the correct range.
		 * Use O(N) linear search here for simplicity. Could be improved
		 * to O(log(N)) binary search, but seems unlikely that a function
		 * would be split into so many subintervals N that this becomes a
		 * serious overhead. That is, using this interpolating class might
		 * not be a good solution in that case anyway.
		 */
		auto it = ranges.cbegin();
		while (it->xmax < x)
			++it;
		return interpolate(x, it->samples, fmin, fmax);
	}

	std::vector<std::pair<real, real>> get_samples() const {
		std::vector<std::pair<real, real>> res(ranges[0].samples.size());
		for (size_t i=0; i<ranges[0].samples.size(); ++i){
			res[i] = std::make_pair(ranges[0].samples[i].x,
			                        ranges[0].samples[i].f);
		}
		return res;
	}


private:
	/*
	 * Structures that contain the sampling points of the interpolated
	 * function:
	 */
	struct xf_t {
		real x;
		real f;

		bool operator==(const xf_t& other) const {
			return (x == other.x) && (f == other.f);
		}
	};
	struct subrange_t {
		real xmax;
		std::vector<xf_t> samples;
	};

	/*
	 * Data members:
	 */
	real xmin;
	real fmin;
	real fmax;
	std::vector<subrange_t> ranges;


//	struct chebyshev_point_t {
//		size_t i;
//		size_t n;
//
//		chebyshev_point_t(size_t i, size_t n) : i(i), n(n)
//		{}
//
//
//	};


	static real chebyshev_point(size_t i, size_t n, real xmin, real xmax){
		constexpr real pi = std::numbers::pi_v<real>;
		const real z = std::cos(i * pi / (n-1));
		return std::min(std::max(xmin + 0.5 * (1.0 + z) * (xmax - xmin), xmin),
		                xmax);
	}

	/*
	 * The routine that refines the Chebyshev support points:
	 */
	template<typename fun_t>
	static std::vector<xf_t>
	refine_chebyshev(fun_t func, real xmin, real xmax,
	                 const std::vector<xf_t>& samples)
	{
		if (samples.empty())
			throw std::runtime_error("Trying to refine empty sample vector.");

		/*
		 * Get the number of intervals.
		 * The old Chebyshev support vector has `m` nodes, and
		 * thereby n_old = m - 1 intervals between the sample points.
		 * We assume that the number of intervals is a power of two.
		 * This means that if we split each interval, we arrive again
		 * at a number of intervals that is a number of two.
		 * Moreover, in the resulting node set {cos(i*pi/(n-1)) : i=0, ..., n},
		 * the old nodes are all contained and we only have to evaluate
		 * the function at the split points of the interval.
		 */
		const size_t m_old = samples.size() - 1;

		/*
		 * Double the number of intervals.
		 */
		const size_t m_new = 2 * m_old;

		/* The refined sample vector: */
		std::vector<xf_t> refined(m_new + 1);

		/*
		 * Now every even element of `refined` (starting with element 0)
		 * can be copied from the old sample vector
		 */
		for (size_t i=0; i<refined.size(); ++i){
			if ((i % 2) == 0){
				refined[i] = samples[i / 2];
			} else {
				const real xi = chebyshev_point(i, refined.size(), xmin, xmax);
				const real fi = func(xi);
				refined[i].x = xi;
				refined[i].f = fi;
			}
		}

		return refined;
	}


	struct error_t {
		real abs = 0.0;
		real rel = 0.0;
		size_t iabs = 0;
		size_t irel = 0;
	};

	static error_t error(real f, real f_ref)
	{
		/* Compute relative and absolute error of this sample
		 * (need some clever trick for relative error when reference
		 *  value is zero).
		 */
		error_t err;
		err.abs = std::abs(f - f_ref);
		if (f_ref == 0){
			if (f != 0)
				err.rel = 1.0;
			else
				err.rel = 0.0;
		} else {
			err.rel = err.abs / std::abs(f_ref);
		}
		return err;
	};


	template<typename fun_t>
	static std::variant<std::vector<xf_t>,std::vector<real>>
	determine_samples(fun_t func, real xmin, real xmax, real tol_rel,
	                  real tol_abs, real fmin, real fmax)
	{
		/* Start with 9 samples: */
		constexpr size_t n_start = 129;
		std::vector<xf_t> samples(n_start);
		for (size_t i=0; i<n_start; ++i){
			const real xi = chebyshev_point(i, n_start, xmin, xmax);
			const real fi = func(xi);
			samples[i].x = xi;
			samples[i].f = fi;
		}

		/* Compute the refined samples: */
		std::vector<xf_t> refined(refine_chebyshev(func, xmin, xmax, samples));

		/* The function that computes the error in interpolation: */
		auto max_error = [fmin, fmax](const std::vector<xf_t>& samples,
		                              const std::vector<xf_t>& refined)
		   -> error_t
		{
			for (size_t i=0; i<refined.size(); ++i){
				if ((i % 2) == 0){
					if (refined[i] != samples[i / 2])
						throw std::runtime_error("refined not correctly copied.");
				} else {

				}
			}

			error_t err({.abs=0.0, .rel=0.0, .iabs=0, .irel=0});
			/*
			 * Each even data point (when first element is index with 1)
			 * is new. These data points are controlled.
			 */
			for (size_t i=1; i<refined.size(); i += 2){
				real fint = interpolate(refined[i].x, samples, fmin, fmax);
				error_t err_i = error(fint, refined[i].f);
				if (err_i.abs > err.abs){
					err.abs = err_i.abs;
					err.iabs = i;
				}
				if (err_i.rel > err.rel){
					err.rel = err_i.rel;
					err.irel = i;
				}
			}
			return err;
		};

		/*
		 * Tolerance criteria as defined by `tol_abs` and `tol_rel`.
		 */
		auto tolerance_fulfilled
		   = [tol_rel, tol_abs](const error_t& err) -> bool
		{
			std::cout << "  err_abs = " << err.abs << ", err_rel = "
			          << err.rel << "\n";
			return (10 * err.abs <= tol_abs) && (10 * err.rel <= tol_rel);
		};

		/*
		 * The main refinement loop.
		 */
//		constexpr unsigned int max_iter = 20; // That is already more than 1e6 samples
		constexpr unsigned int max_iter = 5;
		unsigned int i = 0;
		error_t err;
		while (!tolerance_fulfilled(err = max_error(samples, refined))
		       && (i++ < max_iter))
		{
			/* TODO: Here we need to test for discontinuities! */
			throw std::runtime_error("TODO: NEED TO TEST FOR DISCONTINUITIES.");

			samples.swap(refined);
			refined = refine_chebyshev(func, xmin, xmax, samples);
			std::cout << "Refined to " << refined.size() << " nodes.\n"
			          << std::flush;
		}

		/*
		 * Now we check whether we have to split the interval into sub-intervals
		 * so as to drastically improve the minimum precision across the whole
		 * interval.
		 */
		std::vector<real> splits;

		/*
		 * Check if there are any minima within the range.
		 * Minima may evade the refinement checks above and can be very hard
		 * to model to desired precision using the barycentric Lagrange
		 * interpolator if they go down to zero (without using an extensive
		 * amount of nodes).
		 * If there are single root-minima, we may be better off to split the
		 * interval there.
		 */
		std::vector<xf_t> minima;
		real fprev, fi, fnext;
		fi = samples[0].f;
		fnext = samples[1].f;
		for (size_t i=1; i<samples.size()-1; ++i)
		{
			fprev = fi;
			fi = fnext;
			fnext = samples[i+1].f;
			if ((fprev > fi) && (fnext > fi))
			{
				/* There is a minimum. */
				constexpr size_t MAX_ITER = 50;
				std::uintmax_t max_iter = MAX_ITER;
				constexpr int bits = std::numeric_limits<real>::digits / 2;
				/* Note: the Chebyshev points are monotonously falling with
				 * index `i`:
				 */
				const real xl = samples[i+1].x;
				const real xr = samples[i-1].x;
				std::pair<real, real> min
				  = boost::math::tools::brent_find_minima(func, xl, xr, bits,
				                                          max_iter);
				if (max_iter >= MAX_ITER)
					throw std::runtime_error("Unable to determine the "
					                         "minimum in "
					                         "BayrcentricLagrangeInterpolator "
					                         "initialization.");
				/*
				 * Check whether the minimum is modeled to sufficient
				 * precision:
				 */
				const real f_int = interpolate(min.first, refined, fmin, fmax);
				if (min.second == 0 ||
				    !tolerance_fulfilled(error(f_int, min.second)))
				{
					splits.push_back(min.first);
				}
				std::cout << "Found minimum f(" << min.first << ") = "
				          << min.second << ".\n"
				          << "  used range: [" << xl << ", " << xr << "]\n"
				          << "  iter: " << max_iter << "\n"
				          << std::flush;
			}
		}

		/*
		 * Identify
		 */

		/*
		 * Finally, return either the refined vector or the splits:
		 */
		if (splits.empty())
			return refined;
		return splits;
	}

	template<typename fun_t>
	static std::vector<subrange_t>
	determine_ranges(fun_t func, real xmin, real xmax, real tol_rel,
	                 real tol_abs, real fmin, real fmax, size_t max_splits)
	{
		std::vector<subrange_t> ranges;

		size_t iter = 0;
		std::stack<real> interval_todo;
		interval_todo.push(xmax);
		while (!interval_todo.empty() && iter < max_splits){
			std::variant<std::vector<xf_t>,std::vector<real>>
			    res = determine_samples(func, xmin, interval_todo.top(),
			                            tol_rel, tol_abs, fmin, fmax);
			if (res.index() == 0){
				/*
				 * Successfully determined the samples for this interval.
				 */
				ranges.emplace_back();
				ranges.back().samples.swap(std::get<0>(res));
				ranges.back().xmax = interval_todo.top();
				/*
				 * Proceed to the next interval on the stack:
				 */
				xmin = interval_todo.top();
				interval_todo.pop();
			} else if (res.index() == 1){
				/*
				 * Split(s) was/were requested.
				 */
				std::sort(std::get<1>(res).begin(), std::get<1>(res).end(),
				          std::greater<real>());
				for (const real& s : std::get<1>(res))
					interval_todo.push(s);
			} else {
				throw std::runtime_error("Variant initialization exception in "
				                         "the BarycentricLagrangeInterpolator "
				                         "initialization.");
			}
			++iter;
		}

		if (!interval_todo.empty())
			throw std::runtime_error("Exceeded maximum number of splits.");

		return ranges;
	}


	/*
	 * The interpolation routine:
	 */
	static real interpolate(real x, const std::vector<xf_t>& samples,
	                        real fmin, real fmax)
	{
		typedef typename BLI_summand<real>::type sum_t;

		if (samples.empty())
			throw std::runtime_error("No samples given for barycentric "
			                         "Lagrange interpolator to interpolate "
			                         "from.");
		auto it = samples.cbegin();
		if (x == it->x)
			return it->f;
		sum_t nom = 0.0;
		sum_t denom = 0.0;
		real wi = 0.5 / (x - it->x);
		nom += wi * it->f;
		denom += wi;
		int sign = -1;
		++it;
		for (size_t i=1; i<samples.size()-1; ++i){
			if (x == it->x)
				return it->f;
			wi = sign * 1.0 / (x - it->x);
			nom += wi * it->f;
			denom += wi;
			sign = -sign;
			++it;
		}
		if (x == it->x)
			return it->f;
		wi = sign * 0.5 / (x - it->x);
		nom += wi * it->f;
		denom += wi;
		return std::max<real>(std::min<real>(nom / denom, fmax), fmin);
	}


};


} // namespace numerics
} // namespace reheatfunq

#endif