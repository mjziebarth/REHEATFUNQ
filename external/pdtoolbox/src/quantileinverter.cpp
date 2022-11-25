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

#include <../include/quantileinverter.hpp>
#include <string>
#include <iostream>

using pdtoolbox::QuantileTreeLeaf;
using pdtoolbox::QuantileTreeBranch;
using pdtoolbox::QuantileInverter;

using pdtoolbox::tolerance_policy;
using pdtoolbox::OR;
using pdtoolbox::AND;

/*
 * Gauss-Kronrod:
 */
#include <boost/math/quadrature/gauss_kronrod.hpp>
using boost::math::quadrature::detail::gauss_kronrod_detail;
using boost::math::quadrature::detail::gauss_detail;

const static std::array<double,8> gk75_abscissa
    = gauss_kronrod_detail<double,15,1>::abscissa();

const static std::array<double,8> k7_weights
    = gauss_kronrod_detail<double,15,1>::weights();

const static std::array<double,4> g5_weights
    = gauss_detail<double,7,1>::weights();


const static std::array<double,16> compute_trapezoid_mass(){
	/* Computes a mass matrix that sums up to 1 and
	 * in which mass[i] denotes the relative length of
	 * the interval between abscissa i-7 and abscissa i-6
	 * (where mass[6] is the interval left of 0 and mass[7]
	 *  the one right of 0). */
	std::array<double,16> mass;
	mass[0] = 0.5 * (1.0 - gk75_abscissa[7]);
	for (uint_fast8_t i=0; i<7; ++i){
		mass[i+1] = 0.5*(gk75_abscissa[7-i] - gk75_abscissa[6-i]);
	}
	for (uint_fast8_t i=7; i<14; ++i){
		mass[i+1] = 0.5*(gk75_abscissa[i-6] - gk75_abscissa[i-7]);
	}
	mass[15] = 0.5 * (1.0 - gk75_abscissa[7]);
	return mass;
};


const static std::array<double,16> k7_trapezoid_mass = compute_trapezoid_mass();



/*
 * Points of the Kronrod abscissa transformed from [-1,1] to [0,1]
 */
static std::array<double,17> compute_scaled_k7_lr_abscissa()
{
	std::array<double,17> scab;
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


const static std::array<double,17> scaled_k7_lr_abscissa
   = compute_scaled_k7_lr_abscissa();

/******************************************************************************
 *                                                                            *
 *                             QuantileTreeLeaf                               *
 *                                                                            *
 ******************************************************************************/


template<uint_fast8_t n>
std::array<double,n>
QuantileTreeLeaf::trapezoidal_quantiles(std::array<double,n>&& y) const
{
	/*
	 * Note: This returns quantiles within the range [0,1], counted
	 *       relative to [xmin,xmax].
	 */

	/* Setup the trapezoidal quantiles: */
	std::array<double,16> trapez;
	double S = 0.0;
	for (uint_fast8_t i=0; i<16; ++i){
		trapez[i] = 0.5 * k7_trapezoid_mass[i] * (gk75_lr[i] + gk75_lr[i+1]);
		S += trapez[i];
	}

	/* Normalize to local quantile: */
	double scale = 1.0 / S;
	for (uint_fast8_t i=0; i<16; ++i){
		trapez[i] *= scale;
	}

	/* Now find the quantiles: */
	std::array<double, n> result;
	for (uint_fast8_t i=0; i<n; ++i){
		/* Find the quantile: */
		double last_q = 0.0;
		for (uint_fast8_t j=0; j<16; ++j){
			double next_q = last_q + trapez[j];
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


QuantileTreeLeaf::QuantileTreeLeaf(std::function<double(double)> pdf,
                                   double xmin, double xmax, bool parallel)
{
	/* Bounds: */
	_xmin = xmin;
	_xmax = xmax;

	/* Determine the coordinates of all points: */
	double xc = 0.5*(xmin+xmax);
	double dx = xmax - xmin;
	std::array<double,17> xi;

	xi[0] = xmin;
	for (uint_fast8_t i=1; i<8; ++i) {
		xi[i] = xc - 0.5 * gk75_abscissa[8-i] * dx;
	}
	xi[8] = xc;
	for (uint_fast8_t i=1; i<8; ++i) {
		xi[8+i] = xc + 0.5 * gk75_abscissa[i] * dx;
	}
	xi[16] = xmax;


	/* Evaluate at all the points
	 * (this restructuring allows for parallel evaluation of the
	 * integral loop - however, does not seem to benefit much,
	 * hence commented out currently).
	 * Note: The OMP part seems to introduce a nasty terminate
	 * call in some cases in case the call to pdf() might throw
	 * an exception.
	 * Maybe the try {} catch {terminate();} part is included
	 * even if parallel == false? */
//	#pragma omp parallel for if(parallel)
	for (uint_fast8_t i=0; i<17; ++i){
		gk75_lr[i] = pdf(xi[i]);
	}


	/* Compute Gauss and Kronrod integrals: */
	_kronrod = compute_kronrod();

	#ifdef PDTOOLBOX_QUANTILEINVERTER_CACHE_GAUSS
	_gauss = compute_gauss();
	#endif
}


QuantileTreeLeaf::QuantileTreeLeaf(std::function<double(double)> pdf,
                                   QuantileTreeLeaf::xf_t left,
                                   QuantileTreeLeaf::xf_t right,
                                   bool parallel)
{
	/* Bounds: */
	_xmin = left.x;
	_xmax = right.x;

	/* Determine the coordinates of all points: */
	double xc = 0.5*(_xmin + _xmax);
	double dx = _xmax - _xmin;
	std::array<double,17> xi;

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


std::array<std::unique_ptr<QuantileTreeLeaf>,3>
QuantileTreeLeaf::split(std::function<double(double)> pdf, double z0,
                        double z1) const
{
	/* The x of splitting: */
	double x0 = (1.0 - z0) * _xmin + z0 * _xmax;
	double x1 = (1.0 - z1) * _xmin + z1 * _xmax;
	xf_t l({_xmin, gk75_lr[0]});
	xf_t c0({x0, pdf(x0)});
	xf_t c1({x1, pdf(x1)});
	xf_t r({_xmax, gk75_lr[16]});

	/* Return the array: */
	return std::array<std::unique_ptr<QuantileTreeLeaf>,3>(
	          {std::make_unique<QuantileTreeLeaf>(pdf, l, c0),
	           std::make_unique<QuantileTreeLeaf>(pdf, c0, c1),
	           std::make_unique<QuantileTreeLeaf>(pdf, c1, r)});
}


double QuantileTreeLeaf::gauss() const
{
	#ifdef PDTOOLBOX_QUANTILEINVERTER_CACHE_GAUSS
	return _gauss;
	#else
	return compute_gauss();
	#endif
}


double QuantileTreeLeaf::kronrod() const
{
	return _kronrod;
}


double QuantileTreeLeaf::xmin() const
{
	return _xmin;
}


double QuantileTreeLeaf::xmax() const
{
	return _xmax;
}

double QuantileTreeLeaf::compute_kronrod() const
{
	/* Gauss-Kronrod quadrature: */
	double I = 0.0;
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
	const double cscale = (_xmax - _xmin) / 2.0;

	return I * cscale;
}

double QuantileTreeLeaf::compute_gauss() const
{
	/* Gauss-Legendre quadrature: */
	double I = 0.0;
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
	const double cscale = (_xmax - _xmin) / 2.0;

	return I * cscale;
}


/******************************************************************************
 *                                                                            *
 *                        QuantileTreeBranch                                  *
 *                                                                            *
 ******************************************************************************/
QuantileTreeBranch::QuantileTreeBranch(std::function<double(double)> pdf,
                                       double xmin, double xmax, double atol,
                                       double rtol, tolerance_policy policy)
   : QuantileTreeBranch(std::make_unique<QuantileTreeLeaf>(pdf, xmin, xmax),
                        pdf, atol, rtol, policy, 0)
{
}


static double compute_threshold(double kronrod, double atol, double rtol,
                                tolerance_policy policy)
{
	if (policy == OR){
		return 0.25 * std::max(std::abs(kronrod * rtol), atol);
	}
	return 0.25 * std::min(std::abs(kronrod * rtol), atol);
}


QuantileTreeBranch::QuantileTreeBranch(std::unique_ptr<QuantileTreeLeaf> leaf_,
                                       std::function<double(double)> pdf,
                                       double atol, double rtol,
                                       tolerance_policy policy,
                                       unsigned int descent)
{
	if (descent > 100)
		throw std::runtime_error("Descending too deeply.");

	/* Grow a leaf: */
	leaf.swap(leaf_);

	/* Check and split: */
	check_and_split(pdf, atol, rtol, policy, descent);

	/* Compute the integral: */
	if (leaf){
		I = leaf->kronrod();
	} else {
		I = left_branch->integral() + central_branch->integral()
		    + right_branch->integral();
	}

	/* The root node should validate and trigger the computations of
	 * offsets: */
	if (descent == 0){
		/* Validation: */
		double delta = std::abs(I - 1.0);
		if (delta > compute_threshold(1.0, atol, rtol, policy)/descent){
			std::string msg("Failed to be normalized to precision!\nTarget:  ");
			msg += std::to_string(compute_threshold(1.0, atol, rtol, policy));
			msg += "\nAchieved:";
			msg += std::to_string(delta);
			msg += "\n";
			throw std::runtime_error(msg);
		}
		/* Compute the offsets: */
		set_offsets(0.0);
	}
}


void QuantileTreeBranch::check_and_split(std::function<double(double)> pdf,
                                         double atol, double rtol,
                                         tolerance_policy policy,
                                         unsigned int descent)
{
	if (!leaf)
		throw std::runtime_error("check_and_split called with nullptr leaf.");

	/* Check whether the leaf is exact enough: */
	double delta_I = std::abs(leaf->gauss() - leaf->kronrod());
	if (delta_I > compute_threshold(leaf->kronrod(), atol, rtol, policy)){
		/* Have to split! */
		std::array<std::unique_ptr<QuantileTreeLeaf>,3>
		    leaves(leaf->split(pdf, 1.0/3.0, 2.0/3.0));

		/* Remove the leaf and add two new branches: */
		leaf.reset();
		std::unique_ptr<QuantileTreeBranch>
		   new_branch = std::make_unique<QuantileTreeBranch>(
		                          std::move(leaves[0]),
		                          pdf, atol, rtol, policy, descent + 1);
		left_branch.swap(new_branch);

		new_branch = std::make_unique<QuantileTreeBranch>(std::move(leaves[1]),
		                          pdf, atol, rtol, policy, descent + 1);
		central_branch.swap(new_branch);

		new_branch = std::make_unique<QuantileTreeBranch>(std::move(leaves[2]),
		                          pdf, atol, rtol, policy, descent + 1);
		right_branch.swap(new_branch);
	}
}


void QuantileTreeBranch::set_offsets(double qoff)
{
	this->qoff = qoff;
	if (!leaf){
		left_branch->set_offsets(qoff);
		qoff += left_branch->I;
		central_branch->set_offsets(qoff);
		qoff += central_branch->I;
		right_branch->set_offsets(qoff);
	}
}


std::pair<const QuantileTreeLeaf&,double>
QuantileTreeBranch::find(double q) const
{
	const QuantileTreeBranch* branch = this;
	double qoff = 0.0;

	while (!branch->leaf){
		/* If we are in the left branch, we do not have to update
		 * qoff (the left boundaries coincide): */
		double qoff_next = qoff + branch->left_branch->I;
		if (qoff_next >= q){
			branch = branch->left_branch.get();
			continue;
		}
		/* Increment the offset and perform the same logic to the
		 * central branch: */
		qoff = qoff_next;
		qoff_next += branch->central_branch->I;
		if (qoff_next >= q){
			branch = branch->central_branch.get();
			continue;
		}
		/* Otherwise, we are in the right branch: */
		qoff = qoff_next;
		branch = branch->right_branch.get();
	}

	return std::pair<const QuantileTreeLeaf&,double>(*(branch->leaf), qoff);
}


double QuantileTreeBranch::integral() const
{
	return I;
}

size_t QuantileTreeBranch::leaves() const
{
	if (leaf)
		return 1;

	size_t n = 0;
	n += left_branch->leaves();
	n += central_branch->leaves();
	n += right_branch->leaves();
	return n;
}



/******************************************************************************
 *                                                                            *
 *                            QuantileInverter                                *
 *                                                                            *
 ******************************************************************************/

QuantileInverter::QuantileInverter(std::function<double(double)> pdf,
                                   double xmin, double xmax, double atol,
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

double QuantileInverter::invert(double q) const
{
	/* Find the leaf: */
	std::pair<const QuantileTreeLeaf&,double> rootleaf(root.find(q));
	std::unique_ptr<QuantileTreeLeaf>
	   leaf(std::make_unique<QuantileTreeLeaf>(rootleaf.first));
	double qoff = rootleaf.second;

	/* Now trisect: */
	size_t steps = 0;
	double dq = leaf->kronrod();
	while (dq > compute_threshold(1.0, atol, rtol, policy)){
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
		double z = (q-qoff) / dq;
		if (z < 0 || z > 1)
			throw std::runtime_error("Unlikely z out of bounds detected.");

		/* Estimate through trapezoidal rule where z-0.05 and z+0.05 lie: */
		double zl = (z <= 0.01) ? 0.5 * z : std::max(z-0.00125, 0.01);
		double zr = (z >= 0.99) ? 0.5 * (z + 1.0) : std::min(z+0.00125, 0.99);
		std::array<double,2>
		    zi(leaf->trapezoidal_quantiles<2>({zl, zr}));

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
		std::array<std::unique_ptr<QuantileTreeLeaf>,3>
		    sq3(leaf->split(pdf, zi[0], zi[1]));

		/* Find the containing: */
		double qoff_next = qoff + sq3[0]->kronrod();
		if (q < qoff_next){
			/* Left interval. */
			leaf.swap(sq3[0]);
			double dq_next = leaf->kronrod();
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
			double dq_next = leaf->kronrod();
			if (dq_next == dq)
				throw std::runtime_error("Did not make progress in quantile "
				                         "inversion iteration.");
			dq = dq_next;
			continue;
		}
		/* Right interval. */
		qoff = qoff_next;
		leaf.swap(sq3[2]);
		double dq_next = leaf->kronrod();
		if (dq_next == dq)
			throw std::runtime_error("Did not make progress in quantile "
			                         "inversion iteration.");
		dq = dq_next;
	}
	return 0.5 * (leaf->xmin() + leaf->xmax());
}


size_t QuantileInverter::leaves() const
{
	return root.leaves();
}
