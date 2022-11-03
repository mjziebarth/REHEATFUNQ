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

#ifndef PDTOOLBOX_QUANTILEINVERTER_HPP
#define PDTOOLBOX_QUANTILEINVERTER_HPP

namespace pdtoolbox {

class QuantileTreeBranch;

class QuantileTreeLeaf {
	friend QuantileTreeBranch;
public:
	QuantileTreeLeaf(std::function<double(double)> pdf, double xmin,
	                 double xmax, bool parallel=false);

	QuantileTreeLeaf(const QuantileTreeLeaf& other) = default;
	QuantileTreeLeaf(QuantileTreeLeaf&& other) = default;

	/* Mostly for internal use: */
	struct xf_t {
		double x;
		double f;
	};
	QuantileTreeLeaf(std::function<double(double)> pdf, xf_t left, xf_t right,
	                 bool parallel=false);

	/* Retrieving the integrals approximations of the G7, K15
	 * scheme: */
	double gauss() const;
	double kronrod() const;

	/* Boundaries: */
	double xmin() const;
	double xmax() const;


	/* Splitting the interval: */
	std::array<std::unique_ptr<QuantileTreeLeaf>,3>
	   split(std::function<double(double)> pdf, double z0, double z1) const;

	/* Estimating quantiles using trapezoidal rule (*local* quantiles
	 * within this integral): */
	template<uint_fast8_t n>
	std::array<double,n> trapezoidal_quantiles(std::array<double,n>&& y) const;


private:
	/* Function evaluations of the Gauss-Kronrod 7-15
	 * (plus the function evaluations at the end): */
	std::array<double,17> gk75_lr;
	double _xmin;
	double _xmax;
	double _kronrod;

	/* We don't typically need to cache the evaluation of the
	 * Gauss-Legendre quadrature since it is used only once
	 * for error estimation:
	 */
	#ifdef PDTOOLBOX_QUANTILEINVERTER_CACHE_GAUSS
	double _gauss;
	#endif

	double compute_gauss() const;
	double compute_kronrod() const;
};


enum tolerance_policy {
	OR, AND
};

class QuantileTreeBranch {
public:
	QuantileTreeBranch(std::function<double(double)> pdf, double xmin,
	                   double xmax, double atol, double rtol,
	                   tolerance_policy policy);

	double integral() const;

	size_t leaves() const;

	/* Find the leaf that contains quantile q: */
	std::pair<const QuantileTreeLeaf&,double> find(double q) const;

	/* Mostly for internal use: */
	QuantileTreeBranch(std::unique_ptr<QuantileTreeLeaf> leaf,
	                   std::function<double(double)> pdf,
	                   double atol, double rtol, tolerance_policy policy,
	                   unsigned int descent);

private:
	/* Each branch can have a maximum of one leaf. */
	std::unique_ptr<QuantileTreeLeaf> leaf;
	std::unique_ptr<QuantileTreeBranch> left_branch;
	std::unique_ptr<QuantileTreeBranch> central_branch;
	std::unique_ptr<QuantileTreeBranch> right_branch;

	/* The integral: */
	double I;

	/* Offset within the CDF: */
	double qoff;

	void check_and_split(std::function<double(double)> pdf, double atol,
	                     double rtol, tolerance_policy policy,
	                     unsigned int descent);

	/* Set the offset of this branch and all child branches. */
	void set_offsets(double qoff);
};


class QuantileInverter {
public:
	QuantileInverter(std::function<double(double)> pdf, double xmin,
	                 double xmax, double atol, double rtol,
	                 tolerance_policy policy);

	double invert(double q) const;

	/*
	 * Diagnostic capabilities:
	 */
	size_t leaves() const;

private:
	QuantileTreeBranch root;
	std::function<double(double)> pdf;
	const double atol, rtol;
	tolerance_policy policy;
};


}

#endif