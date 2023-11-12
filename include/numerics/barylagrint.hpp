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
#include <numbers>
#include <utility>
#include <variant>
#include <optional>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <thread>


#include <boost/math/tools/minima.hpp>

#include <numerics/intervall.hpp>
#include <numerics/kahan.hpp>

namespace reheatfunq {
namespace numerics {

/*
 * Types for summation within the BarycentricLagrangeInterpolator
 * so as not to lose precision:
 */
template<typename real>
struct BLI_summand {
	typedef KahanAdder<real> type;
};

template<>
struct BLI_summand<double>{
	typedef long double type;
};

template<>
struct BLI_summand<float>{
	typedef double type;
};


/*
 * Helper exception class:
 */
template<typename real>
class RefinementError : public std::runtime_error {
public:
	RefinementError(const char* c) : std::runtime_error(c)
	{}
};


/*
 * Tolerance Criteria:
 */
template<typename real>
struct AbsoluteOrRelative {
	static bool eval(const real err_abs, const real err_rel,
	                 const real tol_abs, const real tol_rel)
	{
		return (err_abs < tol_abs) || (err_rel < tol_rel);
	}
};


/*
 * The main class:
 */

template<typename real, typename tolerance_success = AbsoluteOrRelative<real>>
class PiecewiseBarycentricLagrangeInterpolator
{
public:
	template<typename fun_t>
	PiecewiseBarycentricLagrangeInterpolator(fun_t func, real xmin, real xmax,
	            real tol_rel = std::sqrt(std::sqrt(std::numeric_limits<real>::epsilon())),
	            real tol_abs = 0.0,
	            real fmin = -std::numeric_limits<real>::infinity(),
	            real fmax = std::numeric_limits<real>::infinity(),
	            size_t max_splits = 100, uint8_t max_refinements = 6)
	   : xmin(xmin), xmax(xmax), fmin(fmin), fmax(fmax),
	     ranges(determine_ranges(func, PointInInterval<real>(xmin, 0, xmax-xmin),
	                             PointInInterval<real>(xmax, xmax-xmin, 0),
	                             tol_rel, tol_abs, fmin, fmax,
	                             max_splits, max_refinements))
	{
		/* This check should never trigger, but let's keep it here to ensure
		 * safety of assuming that ranges is not empty.
		 */
		if (ranges.empty())
			throw std::runtime_error("Ended up with an empty sample range in "
			                         "BarycentricLagrangeInterpolator.");
	}

	template<typename fun_t>
	PiecewiseBarycentricLagrangeInterpolator(
	         PiecewiseBarycentricLagrangeInterpolator&& other)
	   : xmin(other.xmin), xmax(other.xmax), fmin(other.fmin), fmax(other.fmax),
	     ranges(std::move(other.ranges))
	{}

	real operator()(PointInInterval<real> x) const {
		if (x < xmin || x > xmax)
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
		PointInInterval<real> xmin_local(xmin, 0, xmax-xmin);
		while (it->xmax < x){
			xmin_local = it->xmax;
			++it;
		}
		const PointInInterval<real> xint(x, x-xmin, xmax-x);
		return interpolate(support_to_chebyshev(xint, xmin_local, it->xmax),
		                   it->samples, fmin, fmax);
	}

	std::vector<std::vector<std::pair<real, real>>> get_samples() const {
		std::vector<std::vector<std::pair<real,real>>> res(ranges.size());
		for (size_t i=0; i<ranges.size(); ++i){
			res[i].resize(ranges[i].samples.size());
			for (size_t j=0; j<ranges[i].samples.size(); ++j){
				res[i][j] = std::make_pair(
					chebyshev_to_support(ranges[i].samples[j].z,
					                     PointInInterval<real>(xmin, 0, xmax-xmin),
					                     PointInInterval<real>(xmax, xmax-xmin, 0)),
				    ranges[i].samples[j].f
				);
			}
		}
		return res;
	}


private:
	/*
	 * Structures that contain the sampling points of the interpolated
	 * function:
	 */
	struct zf_t {
		PointInInterval<real> z;
		real f;

		bool operator==(const zf_t& other) const {
			return (z == other.z) && (f == other.f);
		}
	};
	struct subrange_t {
		PointInInterval<real> xmax;
		std::vector<zf_t> samples;
	};

	/*
	 * Data members:
	 */
	real xmin;
	real xmax;
	real fmin;
	real fmax;
	std::vector<subrange_t> ranges;


	constexpr static PointInInterval<real> chebyshev_point(size_t i, size_t n){
		constexpr real pi = std::numbers::pi_v<real>;
		const real z = std::cos(i * pi / (n-1));
		const real cha = std::cos(i * pi / (2*(n-1)));
		const real sha = std::sin(i * pi / (2*(n-1)));
		const real zf = cha * cha;
		const real zb = sha * sha;
		return PointInInterval<real>(z, zf, zb);
	}

	static PointInInterval<real>
	chebyshev_to_support(const PointInInterval<real>& z,
	                     const PointInInterval<real>& xmin,
	                     const PointInInterval<real>& xmax)
	{
		const real Dx = xmax - xmin;
		return PointInInterval<real>(
		    std::min<real>(std::max<real>(
		        xmin + 0.5 * (1.0 + z.val) * Dx,
		        xmin),
		    xmax),
		    Dx * z.from_front + xmin.from_front,
		    Dx * z.from_back + xmax.from_back
		);
	}

	static PointInInterval<real>
	chebyshev_to_support(PointInInterval<real>&& z,
	                     const PointInInterval<real>& xmin,
	                     const PointInInterval<real>& xmax){
		const real Dx = xmax - xmin;
		z.val = std::min<real>(std::max<real>(
		        xmin + 0.5 * (1.0 + z.val) * Dx,
		        xmin),
		    xmax);
		z.from_front *= Dx;
		z.from_back *= Dx;
		z.from_front += xmin.from_front;
		z.from_back += xmax.from_back;
		return std::move(z);
	}

	static PointInInterval<real>
	support_to_chebyshev(const PointInInterval<real>& x,
	                     const PointInInterval<real>& xmin,
						 const PointInInterval<real>& xmax){
		const real Dx = xmax - xmin;
		return PointInInterval<real>(
			std::min<real>(std::max<real>(
				2*(x - xmin) / (xmax - xmin) - 1.0,
				-1),
			1),
		    std::min<real>((x.from_front - xmin.from_front) / Dx, 1.0),
		    std::min<real>((x.from_back - xmax.from_back) / Dx, 1.0)
		);
	}

	/*
	 * The routine that refines the Chebyshev support points:
	 */
	template<typename fun_t>
	static std::optional<std::vector<zf_t>>
	refine_chebyshev(fun_t func, const PointInInterval<real>& xmin,
	                 const PointInInterval<real>& xmax,
	                 const std::vector<zf_t>& samples)
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
		std::vector<zf_t> refined(m_new + 1);

		/*
		 * Now every even element of `refined` (starting with element 0)
		 * can be copied from the old sample vector
		 */
		for (size_t i=0; i<refined.size(); ++i){
			if ((i % 2) == 0){
				refined[i] = samples[i / 2];
			} else {
				const PointInInterval<real> zi = chebyshev_point(i, refined.size());
				const real fi = func(chebyshev_to_support(zi, xmin, xmax));
				refined[i].z = zi;
				refined[i].f = fi;
			}
		}

		/* Check consistency: */
		if (refined.size() > 2){
			const real eps = std::numeric_limits<real>::epsilon();
			PointInInterval<real> xlast = chebyshev_to_support(refined[0].z, xmin, xmax);
			for (size_t i=1; i<refined.size(); ++i){
				PointInInterval<real> xi = chebyshev_to_support(refined[i].z, xmin, xmax);
				const real xcmp = std::max<real>(
					std::fabs(xlast),
					std::fabs(xi)
				);
				if (xlast.distance(xi) <= 4 * eps * xcmp){
					return std::optional<std::vector<zf_t>>();
				}
				xlast = xi;
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
	                   const size_t n_chebyshev, const PointInInterval<real>& xmin,
	                   const PointInInterval<real>& xmax, const real fmin, const real fmax)
	{
		/* Add the initial interval as to do: */
		struct interval_t {
			size_t il = 0;
			PointInInterval<real> xl;
			PointInInterval<real> xr;
			real fl = 0;
			real fr = 0;

			interval_t() = default;

			interval_t(size_t il, PointInInterval<real> xl, PointInInterval<real> xr,
			           real fl, real fr)
			   : il(il), xl(xl), xr(xr), fl(fl), fr(fr) {};
		};
		std::stack<interval_t> todo;
		std::stack<interval_t> next_todo;
		std::stack<discontinuity_t> identified;
		{
			PointInInterval<real> xl(
			    chebyshev_to_support(
			        chebyshev_point(il, n_chebyshev),
			        xmin, xmax)
			);
			PointInInterval<real> xr(
			    chebyshev_to_support(
			        chebyshev_point(il+1, n_chebyshev),
			        xmin, xmax)
			);
			PointInInterval<real> xm(
			    chebyshev_to_support(
			        chebyshev_point(2*il+1, 2*(n_chebyshev-1)+1),
			        xmin, xmax)
			);
			if (xl == xm || xr == xm)
				throw RefinementError<real>("Precision of `x` data point exceeded."
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
				auto x = chebyshev_to_support(chebyshev_point(ix, n_cheb_i), xmin, xmax);
				if (x == todo.top().xl || x == todo.top().xr){
//					/* Identify the discontinuity closer than what might be
//					 * possible by the Chebyshev grid evaluation:
//					 */
//					const bool order = todo.top().xl < todo.top().xr;
//					x = todo.top().xl;
//					PointInInterval<real> xnext(
//					    std::nextafter(x, todo.top().xr);
//					while ((xnext < todo.top().xr) == order &&
//					       std::fabs(todo.top().fl - func(xnext)) < 0.5 * df_0)
//					{
//						x = xnext;
//					}
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


	template<typename fun_t, size_t n_start=5, size_t max_recursion_depth=9>
	static std::pair<error_t, std::vector<subrange_t>>
	initial_divide_and_conquer(fun_t func, PointInInterval<real> xmin,
	                  PointInInterval<real> xmax, real f_xmin, real f_xmax,
	                  real tol_rel, real tol_abs, real fmin, real fmax,
	                  size_t recursion_depth = 0)
	{
		static_assert(n_start >= 3);
		static_assert(chebyshev_point(0, 2) > chebyshev_point(1,2));

		/* Start with `n_start` samples: */
		std::vector<zf_t> samples(n_start);
		samples.front().z = chebyshev_point(0, n_start);
		samples.front().f = f_xmax;
		for (size_t i=1; i<n_start-1; ++i){
			const PointInInterval<real> zi(chebyshev_point(i, n_start));
			const real fi = func(chebyshev_to_support(zi, xmin, xmax));
			samples[i].z = zi;
			samples[i].f = fi;
		}
		samples.back().z = chebyshev_point(n_start-1, n_start);
		samples.back().f = f_xmin;

		/* The function that computes the error in interpolation: */
		auto estimate_errors = [fmin, fmax](const std::vector<zf_t>& samples,
		                                    const std::vector<zf_t>& refined)
		   -> std::vector<error_t>
		{
			std::vector<error_t> errors((refined.size()-1) / 2);
			/*
			 * Each even data point (when first element is index with 1)
			 * is new. These data points are controlled.
			 */
			auto it = errors.begin();
			for (size_t i=1; i<refined.size(); i += 2){
				real fint = interpolate(refined[i].z, samples, fmin, fmax);
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


		/* Compute the refined samples: */
		std::optional<std::vector<zf_t>> refined(refine_chebyshev(func, xmin, xmax, samples));
		if (!refined){
			/* Accept this interval. */
			std::vector<subrange_t> ranges;
			ranges.emplace_back(xmax);
			ranges.back().samples.swap(samples);
			return std::make_pair(error_t(), std::move(ranges));
		}

		/*
		 * Compute the error:
		 */
		std::vector<error_t> error(estimate_errors(samples, *refined));
		error_t err(max_error(error));

		/*
		 * Compute the location of the median error:
		 */
		real cumul_err_rel = 0.0;
		for (const error_t& err : error){
			cumul_err_rel += err.rel;
		}
		real cer_i = 0.0;
		size_t i=0;
		PointInInterval<real> xmed; // Something invalid.
		size_t r_med = refined->size();
		for (const error_t& err_i : error){
			cer_i += err_i.rel;
			if (cer_i >= cumul_err_rel / 2){
				/* Add the center of that interval as split: */
				r_med = 2*i + 1;
				xmed = chebyshev_to_support((*refined)[r_med].z, xmin, xmax);
				break;
			}
			++i;
		}


		/*
		 * Check if the median is centered:
		 */
		constexpr double ASYM = 0.2;
		static_assert((ASYM < 0.5) && (ASYM > 0));
		bool centered = ((1-ASYM)*xmin + ASYM * xmax) < xmed
		                 && (ASYM*xmin + (1-ASYM)*xmax) > xmed;
		
		/*
		 * Now compute the subrange(s):
		 */
		std::vector<subrange_t> ranges;
		if (centered || recursion_depth > max_recursion_depth)
		{
			/* Accept this interval. */
			ranges.emplace_back(xmax);
			ranges.back().samples.swap(samples);
		} else {
			/* Divide: */
			real f_xmin = samples.back().f;
			real f_xmed = (*refined)[r_med].f;
			real f_xmax = samples.front().f;
			std::pair<error_t, std::vector<subrange_t>>
			    res0(initial_divide_and_conquer(
			             func, xmin, xmed, f_xmin, f_xmed,
			             tol_rel, tol_abs, fmin, fmax,
			             recursion_depth + 1
			    ));
			std::pair<error_t, std::vector<subrange_t>>
			    res1(initial_divide_and_conquer(
			             func, xmed, xmax, f_xmed,
			             f_xmax, tol_rel,
			             tol_abs, fmin, fmax,
			             recursion_depth + 1
			    ));
			ranges.swap(res0.second);
			ranges.insert(ranges.end(), res1.second.cbegin(), res1.second.cend());
			err.rel = std::max(res0.first.rel, res1.first.rel);
			err.abs = std::max(res0.first.abs, res1.first.abs);
		}
		
		return std::make_pair(err, std::move(ranges));
	}


	template<typename fun_t>
	static std::vector<subrange_t>
	determine_ranges(fun_t func, PointInInterval<real> xmin, PointInInterval<real> xmax,
	                 real tol_rel, real tol_abs, real fmin, real fmax, size_t max_splits,
	                 uint8_t max_refinements)
	{
		std::pair<error_t, std::vector<subrange_t>>
		    initial(initial_divide_and_conquer(
		        func, xmin, xmax,
		        func(xmin), func(xmax),
		        tol_rel, tol_abs, fmin, fmax
		    ));

		std::vector<subrange_t> ranges;
		ranges.swap(initial.second);

		/*
		 * Functions for refinement:
		 */


		/* The function that computes the error in interpolation: */
		auto estimate_errors = [fmin, fmax](const std::vector<zf_t>& samples,
											const std::vector<zf_t>& refined)
		-> std::vector<error_t>
		{
			std::vector<error_t> errors((refined.size()-1) / 2);
			/*
			* Each even data point (when first element is index with 1)
			* is new. These data points are controlled.
			*/
			auto it = errors.begin();
			for (size_t i=1; i<refined.size(); i += 2){
				real fint = interpolate(refined[i].z, samples, fmin, fmax);
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
			return tolerance_success::eval(err.abs, err.rel, tol_abs, tol_rel);
		};


		/* Now refine: */
		PointInInterval<real> xmin_i;
		PointInInterval<real> xmax_i = xmin;
		for (subrange_t& range : ranges){
			xmin_i = xmax_i;
			xmax_i = range.xmax;
			/* Compute the refined samples: */
			std::optional<std::vector<zf_t>> 
			    refined(refine_chebyshev(func, xmin_i, xmax_i, range.samples));

			/* If no refinements possible, continue: */
			if (!refined)
				continue;

			uint_fast8_t iter = 0;
			std::vector<error_t> new_err;
			std::vector<error_t> old_err;
			error_t err;
			while (!tolerance_fulfilled(
			           err = max_error(new_err = estimate_errors(range.samples, *refined))
			          )
			       && (iter++ < max_refinements))
			{
				/* Test for discontinuities.
				* The discontinuity check uses the property that a discontinuity
				* stays constant throughout refinement. In contrast, any smooth
				* part will eventually continuously reduce its error with
				* increased sampling.
				*/
				if (!old_err.empty()){
					/*  */
					std::vector<discontinuity_t> discontinuities;
					for (size_t i=0; i<new_err.size(); ++i){
						/* Restrict these checks to those intervals in which the
						* tolerance criteria is not fulfilled:
						*/
						if (!tolerance_fulfilled(new_err[i])){
							if (new_err[i].abs > 0.9 * old_err[i / 2].abs)
							{
								/* Test for the discontinuity. */
								try {
									std::optional<discontinuity_t> disco(
										find_discontinuity(func, i, new_err.size()+1,
														xmin_i, xmax_i, fmin, fmax)
									);
									if (disco && disco->x != xmin_i && disco->x != xmax_i){
										/* Is a discontinuity! */
										discontinuities.push_back(*disco);
									}
								} catch (RefinementError<real>& re) {
								}
							}
						}
					}
					if (!discontinuities.empty()){
						std::cerr << "Found discontinuity but do not know what to do.\n";
						//return discontinuities;
					}
				}

				/* Refinement: */
				old_err.swap(new_err);
				range.samples.swap(*refined);
				refined = refine_chebyshev(func, xmin_i, xmax_i, range.samples);

				if (!refined)
					break;
			}
		}

		return ranges;
	}


	/*
	 * The interpolation routine:
	 */
	static real
	interpolate(
	    PointInInterval<real> z,
	    const std::vector<zf_t>& samples,
	    real fmin, real fmax
	)
	{
		typedef typename BLI_summand<real>::type sum_t;

		if (samples.empty())
			throw std::runtime_error("No samples given for barycentric "
			                         "Lagrange interpolator to interpolate "
			                         "from.");
		auto it = samples.cbegin();
		if (z.val == it->z.val)
			return it->f;
		sum_t nom = 0.0;
		sum_t denom = 0.0;
		real wi = 0.5 / (z - it->z);
		nom += wi * it->f;
		denom += wi;
		int sign = -1;
		++it;
		for (size_t i=1; i<samples.size()-1; ++i){
			if (z.val == it->z.val)
				return it->f;
			wi = sign * 1.0 / (z - it->z);
			nom += wi * it->f;
			denom += wi;
			sign = -sign;
			++it;
		}
		if (z == it->z)
			return it->f;
		wi = sign * 0.5 / (z - it->z);
		nom += wi * it->f;
		denom += wi;
		return std::max<real>(std::min<real>(nom / denom, fmax), fmin);
	}


};


} // namespace numerics
} // namespace reheatfunq

#endif