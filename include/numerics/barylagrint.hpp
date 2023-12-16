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
#include <numerics/functions.hpp>

namespace reheatfunq {
namespace numerics {

namespace rm = reheatfunq::math;

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
	            real tol_rel = rm::sqrt(rm::sqrt(std::numeric_limits<real>::epsilon())),
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
		while (it->xmax < static_cast<real>(x)){
			xmin_local = it->xmax;
			++it;
		}
		const PointInInterval<real> xint(static_cast<real>(x),
		                                 static_cast<real>(x)-xmin,
		                                 xmax-static_cast<real>(x));
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

	void get_samples(std::vector<std::vector<std::pair<double,double>>>& samples) const
	{
		samples.resize(ranges.size());
		auto it = samples.begin();
		PointInInterval<real> xmin_local(xmin, 0.0, xmax-xmin);
		for (const subrange_t& sr : ranges){
			it->resize(sr.samples.size());
			for (size_t i=0; i<sr.samples.size(); ++i){
				(*it)[i].first = static_cast<double>(
					chebyshev_to_support(
					    sr.samples[i].z,
					    xmin_local,
					    sr.xmax
					).val
				);
				(*it)[i].second = static_cast<double>(sr.samples[i].f);
			}
			xmin_local = sr.xmax;
			++it;
		}
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


	template<typename R=real>
	constexpr static
	std::enable_if_t<(std::is_same_v<R, double>
	                 || std::is_same_v<R, long double>),
	PointInInterval<R>>
	chebyshev_point(size_t i, size_t n){
		constexpr R pi = std::numbers::pi_v<R>;
		const R z = rm::cos(i * pi / (n-1));
		const R cha = rm::cos(i * pi / (2*(n-1)));
		const R sha = rm::sin(i * pi / (2*(n-1)));
		const R zf = cha * cha;
		const R zb = sha * sha;
		return PointInInterval<R>(z, zf, zb);
	}


	template<typename R=real>
	static
	std::enable_if_t<!(std::is_same_v<R,double>
	                 || std::is_same_v<R, long double>),
	PointInInterval<R>>
	chebyshev_point(size_t i, size_t n){
		const R pi = boost::math::constants::pi<R>();
		const R z = rm::cos(i * pi / (n-1));
		const R cha = rm::cos(i * pi / (2*(n-1)));
		const R sha = rm::sin(i * pi / (2*(n-1)));
		const R zf = cha * cha;
		const R zb = sha * sha;
		return PointInInterval<R>(z, zf, zb);
	}

	static PointInInterval<real>
	chebyshev_to_support(const PointInInterval<real>& z,
	                     const PointInInterval<real>& xmin,
	                     const PointInInterval<real>& xmax)
	{
		const real Dx = xmax - xmin;
		return PointInInterval<real>(
		    std::min<real>(std::max<real>(
		        xmin.val + 0.5 * (1.0 + z.val) * Dx,
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
		        xmin.val + 0.5 * (1.0 + z.val) * Dx,
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
				2*(x - xmin) / Dx - 1.0,
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
				if (rm::isnan(fi)){
					std::string msg("NaN function evaluation in barycentric Lagrange "
					                "interpolator (at x = ");
					msg += std::to_string(static_cast<long double>(
					            chebyshev_to_support(zi, xmin, xmax).val
					));
					msg += ", z = ";
					msg += std::to_string(static_cast<long double>(zi.val));
					msg += ", xmin = ";
					msg += std::to_string(static_cast<long double>(xmin.val));
					msg += ", xmax = ";
					msg += std::to_string(static_cast<long double>(xmax.val));
					msg += ").";
					throw std::runtime_error(msg);
				} else if (rm::isinf(fi)) {
					std::string msg("INF function evaluation in barycentric Lagrange "
					                "interpolator (at x=");
					msg += std::to_string(static_cast<long double>(
					            chebyshev_to_support(zi, xmin, xmax).val
					));
					msg += ").";
					throw std::runtime_error(msg);
				}
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
					rm::abs(static_cast<real>(xlast)),
					rm::abs(static_cast<real>(xi))
				);
				if (xlast.distance(xi) <= 4 * eps * xcmp){
					return std::optional<std::vector<zf_t>>();
				}
				xlast = xi;
			}
		}

		return refined;
	}


	static void repair_boundary_infinities(std::vector<zf_t>& samples,
	                                       std::vector<zf_t>& refined,
	                                       const real fmin, const real fmax)
	{
		using boost::math::tools::brent_find_minima;

		if (samples.size() < 2)
			throw std::runtime_error("samples empty in repair_boundary_infinities.");

		/*
		 * Now we try to fix infinities assuming that the function
		 * to be interpolated is the logarithm of a relevant property.
		 * Then, we can replace infinity with some small value that
		 * is small enough to exceed floating point precision when
		 * exponentiated (e.g. 1e-400 for double precision).
		 */


		/* Collect all infinities: */
		bool start_inf = rm::isinf(samples.front().f);
		bool end_inf = rm::isinf(samples.back().f);
		for (size_t i=1; i<refined.size()-1; ++i){
			if (rm::isinf(refined[i].f))
				throw std::runtime_error("Infinity outside boundary. Cannot repair.");
		}
		if (start_inf && samples.front().f > fmin)
				throw std::runtime_error("Unfixable positive infinity.");
		if (end_inf && samples.back().f > fmin)
				throw std::runtime_error("Unfixable positive infinity.");


		/* Compute the maximum inter-sample delta f: */
		real df_max = 0.0;
		for (size_t i=1; i<refined.size()-2; ++i){
			df_max = std::max<real>(
				df_max,
				rm::abs(refined[i].f - refined[i+1].f)
			);
		}

		/* Try to fix: */
		auto cost = [fmin, fmax, start_inf, end_inf,
		             &samples, &refined]
		            (real df_start, real df_end)
		-> real
		{
			/* Set the corrected values: */
			if (start_inf)
				samples.front().f = refined[1].f - df_start;
			if (end_inf)
				samples.back().f = refined[refined.size()-2].f - df_end;


			/*
			 * Each odd data point is new. These data points are controlled.
			 */
			real err_abs2 = 0.0;
			for (size_t i=1; i<refined.size(); i += 2){
				real fint = interpolate(refined[i].z, samples, fmin, fmax);
				error_t err_i(error(fint, refined[i].f));
				err_abs2 += err_i.abs * err_i.abs;
			}

			return err_abs2;
		};

		/* Use Powell's method to optimze df_start and df_end: */
		real df_start = 0.0;
		real df_end = 0.0;
		const real tol = boost::math::tools::root_epsilon<real>();
		for (size_t step=0; step<100; ++step){
			real delta_df_start = 0.0;
			real delta_df_end = 0.0;
			if (start_inf){
				auto cost_start = [df_end, &cost](real df_start) -> real
				{
					return cost(df_start, df_end);
				};
				std::uintmax_t max_iter = 40;
				std::pair<real,real>
				dfmin = brent_find_minima(
				            cost_start,
				            static_cast<real>(0.0),
				            static_cast<real>(2*df_max),
				            10,
				            max_iter
				);
				delta_df_start = rm::abs(dfmin.first - df_start);
				df_start = dfmin.first;
			}
			if (end_inf){
				auto cost_end = [df_start, &cost](real df_end) -> real
				{
					return cost(df_start, df_end);
				};
				std::uintmax_t max_iter = 40;
				std::pair<real,real>
				dfmin = brent_find_minima(
				            cost_end,
				            static_cast<real>(0.0),
				            static_cast<real>(2*df_max),
				            10,
				            max_iter
				);
				delta_df_end = rm::abs(dfmin.first - df_end);
				df_end = dfmin.first;
			}
			if (!start_inf || !end_inf || std::max(delta_df_start, delta_df_end) < tol)
				break;
		}

		/* Apply the fix: */
		if (start_inf)
			samples.front().f = refined[1].f - df_start;
		if (end_inf)
			samples.back().f = refined[refined.size()-2].f - df_end;
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
		err.abs = rm::abs(f - f_ref);
		if (f_ref == 0){
			if (f != 0)
				err.rel = 1.0;
			else
				err.rel = 0.0;
		} else {
			err.rel = err.abs / rm::abs(f_ref);
		}
		if (rm::isnan(err.abs)){
			err.abs = std::numeric_limits<real>::infinity();
			err.rel = std::numeric_limits<real>::infinity();
		}
		return err;
	}


	template<typename fun_t, size_t n_start=9, size_t max_recursion_depth=4>
	static std::vector<subrange_t>
	initial_range(fun_t func, PointInInterval<real> xmin,
	              PointInInterval<real> xmax, real f_xmin, real f_xmax,
	              real tol_rel, real tol_abs, real fmin, real fmax,
	              uint8_t max_refinements,
	              size_t recursion_depth = 0)
	{
		static_assert(n_start >= 3);
		static_assert(chebyshev_point<double>(0, 2) > chebyshev_point<double>(1,2));

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

		std::vector<subrange_t> ranges;
		ranges.emplace_back(xmax);
		ranges.back().samples.swap(samples);

		return ranges;
	}


	template<typename fun_t, bool repair_infinity=true>
	static std::vector<subrange_t>
	determine_ranges(fun_t func, PointInInterval<real> xmin, PointInInterval<real> xmax,
	                 real tol_rel, real tol_abs, real fmin, real fmax, size_t max_splits,
	                 uint8_t max_refinements)
	{
		std::vector<subrange_t>
		ranges(initial_range(
		       func, xmin, xmax,
		       func(xmin), func(xmax),
		       tol_rel, tol_abs, fmin, fmax,
		       max_refinements
		));

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
		sum_t nom(0.0);
		sum_t denom(0.0);
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

		real result = std::max<real>(
			std::min<real>(
				static_cast<real>(nom) / static_cast<real>(denom),
				fmax),
			fmin
		);

		return result;
	}


};


} // namespace numerics
} // namespace reheatfunq

#endif