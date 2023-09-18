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
#include <optional>
#include <iostream>
#include <iomanip>
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
	            size_t max_splits = 10, unsigned char max_refinements = 3)
	   : xmin(xmin), fmin(fmin), fmax(fmax),
	     ranges(determine_ranges(func, xmin, xmax, tol_rel, tol_abs, fmin,
	                             fmax, max_splits, max_refinements))
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
	};

	static error_t error(real f, real f_ref)
	{
		/* Compute relative and absolute error of this sample
		 * (need some clever trick for relative error when reference
		 *  value is zero).
		 */
		error_t err;
		err.abs = std::fabs(f - f_ref);
		if (f_ref == 0){
			if (f != 0)
				err.rel = 1.0;
			else
				err.rel = 0.0;
		} else {
			err.rel = err.abs / std::fabs(f_ref);
		}
		return err;
	}

	struct discontinuity_t {
		real x;

		bool operator>(const discontinuity_t& other) const {
			return x > other.x;
		}
	};

	/*
	 * This function tries to identify a possible discontinuity in an interval
	 * range. The discontinuity needs to accumulate more than 50% of the
	 * function value change in the interval.
	 * Note that it tries to identify exactly one discontinuity. If there is
	 * more than one, it may not work or it may return the one that accumulates
	 * more than 50% of the function change in the interval.
	 */
	template<typename fun_t>
	static std::optional<discontinuity_t>
	find_discontinuity(fun_t func, const size_t il,
	                   const size_t n_chebyshev, const real xmin,
	                   const real xmax, const real fmin, const real fmax)
	{
		/* Add the initial interval as to do: */
		struct interval_t {
			size_t il = 0;
			real xl = 0;
			real xr = 0;
			real fl = 0;
			real fr = 0;

			interval_t() = default;

			interval_t(size_t il, real xl, real xr, real fl, real fr)
			   : il(il), xl(xl), xr(xr), fl(fl), fr(fr) {};
		};
		std::stack<interval_t> todo;
		std::stack<interval_t> next_todo;
		std::stack<discontinuity_t> identified;
		{
			real xl = chebyshev_point(il, n_chebyshev, xmin, xmax);
			real xr = chebyshev_point(il+1, n_chebyshev, xmin, xmax);
			real xm = chebyshev_point(2*il+1, 2*(n_chebyshev-1)+1, xmin, xmax);
			if (xl == xm || xr == xm)
				throw std::runtime_error("Precision of `x` data point exceeded."
				                         " Cannot perform required refinement "
				                         "in find_discontinuity.");
			todo.emplace(il, xl, xr, func(xl), func(xr));
		}

		/* Start the refinement loop: */
		const real df_0 = std::fabs(todo.top().fl - todo.top().fr);
		const size_t max_refinements = std::numeric_limits<real>::digits;
		size_t n_cheb_i = n_chebyshev;
		bool exit = false;
		for (uint_fast8_t r=0; r<max_refinements; ++r){
			n_cheb_i = 2 * (n_cheb_i - 1) + 1;
			while (!todo.empty()){
				/* Split the interval: */
				const size_t ix = 2 * todo.top().il + 1;
				auto x = chebyshev_point(ix, n_cheb_i, xmin, xmax);
				if (x == todo.top().xl || x == todo.top().xr){
					/* Identify the discontinuity closer than what might be
					 * possible by the Chebyshev grid evaluation:
					 */
					const bool order = todo.top().xl < todo.top().xr;
					x = todo.top().xl;
					real xnext = std::nextafter(x, todo.top().xr);
					while ((xnext < todo.top().xr) == order &&
					       std::fabs(todo.top().fl - func(xnext)) < 0.5 * df_0)
					{
						x = xnext;
					}
					identified.push(discontinuity_t(x));
				} else {
					const real fx = func(x);

					/* Now check the subintervals for discontinuities: */
					if (std::fabs(todo.top().fl - fx) > 0.5 * df_0){
						next_todo.emplace(2 * todo.top().il, todo.top().xl,
						                  x, todo.top().fl, fx);
					}
					if (std::fabs(todo.top().fr - fx) > 0.5 * df_0){
						next_todo.emplace(ix, x, todo.top().xr,
						                  fx, todo.top().fr);
					}
				}
				todo.pop();
			}

			/* New todo: */
			todo.swap(next_todo);

			if (exit || todo.empty()){
				break;
			}

		}
		/*
		 * Note: If there are multiple discontinuities in the interval, the
		 * resolution of the sampling is probably not great enough since
		 * it implies more than 100% of df_0 allocated in sub-intervals
		 * (i.e. non-monotony).
		 */
		if (identified.size() == 1){
			return identified.top();
		}
		return std::optional<discontinuity_t>();
	}

	template<typename fun_t>
	static std::variant<std::vector<xf_t>,std::vector<real>,
	                    std::vector<discontinuity_t>>
	determine_samples(fun_t func, real xmin, real xmax, real tol_rel,
	                  real tol_abs, real fmin, real fmax,
	                  unsigned char max_iter)
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
		auto estimate_errors = [fmin, fmax](const std::vector<xf_t>& samples,
		                                    const std::vector<xf_t>& refined)
		   -> std::vector<error_t>
		{
			std::vector<error_t> errors((refined.size()-1) / 2);
			/*
			 * Each even data point (when first element is index with 1)
			 * is new. These data points are controlled.
			 */
			auto it = errors.begin();
			for (size_t i=1; i<refined.size(); i += 2){
				real fint = interpolate(refined[i].x, samples, fmin, fmax);
				*it = error(fint, refined[i].f);
				++it;
			}
			return errors;
		};

		auto max_error = [](const std::vector<error_t>& errors) -> error_t
		{
			error_t err({.abs=0.0, .rel=0.0});
			for (const error_t& err_i : errors){
				if (err_i.abs > err.abs){
					err.abs = err_i.abs;
				}
				if (err_i.rel > err.rel){
					err.rel = err_i.rel;
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
			return (10 * err.abs <= tol_abs) && (10 * err.rel <= tol_rel);
		};

		/*
		 * The main refinement loop.
		 */
//		constexpr unsigned int max_iter = 20; // That is already more than 1e6 samples
		unsigned int iter = 0;
		error_t err;
		std::vector<error_t> new_err;
		std::vector<error_t> old_err;
		while (!tolerance_fulfilled(
		           err = max_error(new_err = estimate_errors(samples, refined))
		          )
		       && (iter++ < max_iter))
		{
			/* Test for discontinuities.
			 * The discontinuity check uses the property that a discontinuity
			 * stays constant throughout refinement. In contrast, any smooth
			 * part will eventually continuously reduce its error with
			 * increased sampling.
			 */
			if (!old_err.empty()){
				/*  */
				std::cout << "old_err.size() : " << old_err.size() << "\n";
				std::cout << "new_err.size() : " << new_err.size() << "\n";
				std::cout << "samples.size() : " << samples.size() << "\n";
				std::cout << std::flush;
				std::vector<discontinuity_t> discontinuities;
				for (size_t i=0; i<new_err.size(); ++i){
					/* Restrict these checks to those intervals in which the
					 * tolerance criteria is not fulfilled:
					 */
					if (!tolerance_fulfilled(new_err[i])){
						if (new_err[i].abs > 0.9 * old_err[i / 2].abs)
						{
							/* Test for the discontinuity. */
							std::optional<discontinuity_t> disco(
							    find_discontinuity(func, i, new_err.size()+1,
							                       xmin, xmax, fmin, fmax)
							);
							if (disco){
								std::cout << "found discontinuity in interval ["
									<< chebyshev_point(i+1, new_err.size()+1,
									                   xmin, xmax)
									<< ", "
									<< chebyshev_point(i, new_err.size()+1,
									                   xmin, xmax)
									<< "]\n";
								/* Is a discontinuity! */
								discontinuities.push_back(*disco);
							}
						}
					}
				}
				if (!discontinuities.empty()){
					std::cout << std::setprecision(16);
					std::cout << "Returning discontinuities [";
					for (const discontinuity_t& d : discontinuities){
						std::cout << d.x << ", ";
					}
					std::cout << "]\n" << std::flush;
					return discontinuities;
				}
			}

			/* Refinement: */
			old_err.swap(new_err);
			samples.swap(refined);
			refined = refine_chebyshev(func, xmin, xmax, samples);
		}

		auto compute_fmax = [fmax](const std::vector<xf_t>& refined) -> real
		{
			real fm = (std::isinf(fmax))
			          ? -std::numeric_limits<real>::infinity()
			          : fmax;
			for (const xf_t& xf : refined){
				if (xf.f > fm)
					fm = xf.f;
			}
			return fm;
		};
		const real fmax_i = compute_fmax(refined);

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
		 * If there are a small number of root-minima, we may be better off to
		 * split the interval there.
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
					/* Find the boundary of another interval in which
					 * the function's minmax variation is lower than
					 * the double range:
					 */
					const real f_up = std::min(
					   1e-5 * min.second / std::numeric_limits<real>::epsilon(),
					   1e5 * fmax_i * std::numeric_limits<real>::epsilon()
					);
					uint_fast8_t successes = 0;
					size_t j0, j1;
					for (j0=i-1; j0>0; --j0){
						if (samples[j0].f >= f_up){
							++successes;
							break;
						}
					}
					for (j1=i+1; j1>0; ++j1){
						if (samples[j1].f >= f_up){
							++successes;
							break;
						}
					}
					if (successes == 2){
						std::cout << "add splits [" << samples[j1].x << ", "
						          << samples[j0].x << "]\n";
						splits.push_back(samples[j0].x);
						splits.push_back(samples[j1].x);
					} else {
						/*
						 * Else just split once:
						 */
						splits.push_back(min.first);
					}
				}
//				std::cout << "Found minimum f(" << min.first << ") = "
//				          << min.second << ".\n"
//				          << "  used range: [" << xl << ", " << xr << "]\n"
//				          << "  iter: " << max_iter << "\n"
//				          << std::flush;
			}
		}

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
	                 real tol_abs, real fmin, real fmax, size_t max_splits,
	                 unsigned char max_refinements)
	{
		std::vector<subrange_t> ranges;

		size_t iter = 0;
		struct interval_t {
			real xmin;
			real xmax;
			interval_t(real xmin, real xmax) : xmin(xmin), xmax(xmax) {};
		};
		std::stack<interval_t> interval_todo;
		interval_todo.emplace(xmin, xmax);
		while (!interval_todo.empty() && iter < max_splits){
			std::cout << std::setprecision(16);
			std::cout << "determine_samples([" << interval_todo.top().xmin
			          << ", " << interval_todo.top().xmax << "])\n";
			std::variant<std::vector<xf_t>,std::vector<real>,
			             std::vector<discontinuity_t>>
			    res = determine_samples(func, interval_todo.top().xmin,
			                            interval_todo.top().xmax,
			                            tol_rel, tol_abs, fmin, fmax,
			                            max_refinements);
			if (res.index() == 0){
				std::cout << "no split!\n";
				/*
				 * Successfully determined the samples for this interval.
				 */
				ranges.emplace_back();
				ranges.back().samples.swap(std::get<0>(res));
				ranges.back().xmax = interval_todo.top().xmax;
				/*
				 * Proceed to the next interval on the stack:
				 */
				interval_todo.pop();
			} else if (res.index() == 1){
				std::cout << "split!\n";
				/*
				 * Split(s) was/were requested.
				 */
				std::sort(std::get<1>(res).begin(), std::get<1>(res).end(),
				          std::greater<real>());
				real xmax_i = interval_todo.top().xmax;
				real xmin_i = interval_todo.top().xmin;
				interval_todo.pop();
				if (xmax_i <= xmin_i)
					throw std::runtime_error("xmax_i <= xmin_i in index 2.");
				for (const real& s : std::get<1>(res)){
					interval_todo.emplace(s, xmax_i);
					xmax_i = s;
				}
				interval_todo.emplace(xmin_i, xmax_i);
			} else if (res.index() == 2){
				std::cout << "discontinuity split!\n";
				std::sort(std::get<2>(res).begin(), std::get<2>(res).end(),
				          std::greater<discontinuity_t>());
				real xmax_i = interval_todo.top().xmax;
				real xmin_i = interval_todo.top().xmin;
				if (xmax_i <= xmin_i)
					throw std::runtime_error("xmax_i <= xmin_i in index 2.");
				interval_todo.pop();
				for (const discontinuity_t& s : std::get<2>(res)){
					interval_todo.emplace(s.x, xmax_i);
					xmax_i = std::nextafter(s.x, xmin_i);
				}
				interval_todo.emplace(xmin_i, xmax_i);
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