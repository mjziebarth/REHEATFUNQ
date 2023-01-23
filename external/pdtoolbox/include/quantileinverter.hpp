/*
 * Quantile inversion.
 * This code uses adaptive 15/7 Gauss-Kronrod quadrature to establish
 * the CDF of a PDF to a desired precision. The interval splitting is
 * performed as a tree with three branches per level.
 * Starting from the initially converged tree, a iteratively refining
 * bracketing search (of logarithmic complexity) is performed to
 * determine quantiles.
 * Each iteration estimates tight bounds (1.25e-3 times the current
 * bracket size) around the quantile location through linear
 * interpolation of the trapezoid integral evaluated at the already
 * computed Kronrod samples of the PDF. The hit rate of this approach
 * increases as the bracket is small enough for a good linear
 * approximation (i.e. works well for smooth PDFs).
 * On a miss, use one of the other intervals (probably leading to an
 * interval halving).
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
#include <utility>

#include <quantiletree/branch.hpp>

#ifndef PDTOOLBOX_QUANTILEINVERTER_HPP
#define PDTOOLBOX_QUANTILEINVERTER_HPP

namespace pdtoolbox {

template<typename real>
class QuantileInverter {
public:
	QuantileInverter(std::function<real(real)> pdf, real xmin,
	                 real xmax, double atol, double rtol,
	                 tolerance_policy policy);

	real invert(real q) const;

	/*
	 * Diagnostic capabilities:
	 */
	size_t leaves() const;

private:
	QuantileTreeBranch<real> root;
	std::function<real(real)> pdf;
	const double atol, rtol;
	tolerance_policy policy;
};


template<typename real>
QuantileInverter<real>::QuantileInverter(std::function<real(real)> pdf,
                                         real xmin, real xmax, double atol,
                                         double rtol, tolerance_policy policy)
   : root(pdf, xmin, xmax, atol, rtol, policy), pdf(pdf), atol(atol),
     rtol(rtol), policy(policy)
{
}


/*
 * The algorithm to determine the quantile should do the following:
 *       1) Traverse the QuantileTreeBranch structure until the
 *          leaf containing the quantile q is found.
 *       2) Starting from that leaf spanning the interval [zl,zr], iterate
 *          the following:
 *          a) If the quantile q is at local quantile ql
 *                   q = I[zl] + ql*(I[zr] - I[zl]),
 *             find within the quantile range of the leaf (= its integral dI)
 *             the quantiles
 *                   z0 = I[zl] + dI * (ql - 0.05),
 *                   z1 = I[zl] + dI * (ql + 0.05)
 *             using the trapezoidal_quantiles function.
 *          b) Split the leaf (within the local iterating function) at z0
 *             and z1.
 *          c) Find of the three split leaves the one that contains q
 *             Most of the time, it should be the middle one (if the trapezoidal
 *             approximation is good). Then, we decrease the bracket to
 *             10% of its size each iteration if we area close to linear CDF
 *             within the bracket.
 *          d) With the containing interval [zc0, zc1], continue from a)
 */
template<typename real>
real QuantileInverter<real>::invert(real q) const
{
	/* Find the leaf: */
	std::pair<const QuantileTreeLeaf<real>&, real> rootleaf(root.find(q));
	std::unique_ptr<QuantileTreeLeaf<real>>
	   leaf(std::make_unique<QuantileTreeLeaf<real>>(rootleaf.first));
	real qoff = rootleaf.second;

	/* Now trisect: */
	size_t steps = 0;
	real dq = leaf->kronrod();
	while (dq > QuantileTreeBranch<real>::compute_threshold(1.0, atol, rtol,
	                                                        policy))
	{
		++steps;
		/* The quantile span of this leaf: */

		/* Sanity check: */
		if (std::isnan(dq))
			throw std::runtime_error("Detected NaN dq in QuantileInverter.");
		if (std::isinf(dq))
			throw std::runtime_error("Detected infinite dq in "
			                         "QuantileInverter.");

		/* Within this range, where does q lie relatively? This is
		 * the local quantile z: */
		real z = (q-qoff) / dq;
		if (z < 0 || z > 1)
			throw std::runtime_error("Unlikely z out of bounds detected.");

		/* Estimate through trapezoidal rule where z-0.05 and z+0.05 lie: */
		real zl = (z <= 0.01) ? 0.5 * z : std::max<real>(z-0.00125, 0.01);
		real zr = (z >= 0.99) ? 0.5 * (z + 1.0)
		                      : std::min<real>(z+0.00125, 0.99);
		std::array<real,2>
		    zi(leaf->template trapezoidal_quantiles<2>({zl, zr}));

		/* Numerics check: */
		if (zi[1] < zi[0]){
			std::string msg("While computing quantile, the bracket flipped.\n"
			                "  z[0]: ");
			msg += std::to_string(zi[0]);
			msg += "\n  z[1]: ";
			msg += std::to_string(zi[1]);
			msg += "\n";
			throw std::runtime_error(msg);
		}

		/* Split at these quantiles: */
		std::array<std::unique_ptr<QuantileTreeLeaf<real>>,3>
		    sq3(leaf->split(pdf, zi[0], zi[1]));

		/* Find the containing: */
		real qoff_next = qoff + sq3[0]->kronrod();
		if (q < qoff_next){
			/* Left interval. */
			leaf.swap(sq3[0]);
			real dq_next = leaf->kronrod();
			if (dq_next == dq)
				throw std::runtime_error("Did not make progress in quantile "
				                         "inversion iteration.");
			dq = dq_next;
			continue;
		}
		qoff = qoff_next;
		qoff_next += sq3[1]->kronrod();
		if (q <= qoff_next){
			/* Central interval. */
			leaf.swap(sq3[1]);
			real dq_next = leaf->kronrod();
			if (dq_next == dq)
				throw std::runtime_error("Did not make progress in quantile "
				                         "inversion iteration.");
			dq = dq_next;
			continue;
		}
		/* Right interval. */
		qoff = qoff_next;
		leaf.swap(sq3[2]);
		real dq_next = leaf->kronrod();
		if (dq_next == dq)
			throw std::runtime_error("Did not make progress in quantile "
			                         "inversion iteration.");
		dq = dq_next;
	}
	return 0.5 * (leaf->xmin() + leaf->xmax());
}


template<typename real>
size_t QuantileInverter<real>::leaves() const
{
	return root.leaves();
}



}

#endif