/*
 * Quantile inversion using an interval halving tree with Gauss-Kronrod
 * quadrature. This file covers the tree branches.
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

#include <quantiletree/leaf.hpp>

#ifndef PDTOOLBOX_QUANTILETREE_BRANCH_HPP
#define PDTOOLBOX_QUANTILETREE_BRANCH_HPP

namespace pdtoolbox {

enum tolerance_policy {
	OR, AND
};

template<typename real>
class QuantileTreeBranch {
public:
	QuantileTreeBranch(std::function<real(real)> pdf, real xmin,
	                   real xmax, double atol, double rtol,
	                   tolerance_policy policy);

	real integral() const;

	size_t leaves() const;

	/* Find the leaf that contains quantile q: */
	std::pair<const QuantileTreeLeaf<real>&,real> find(real q) const;

	/* Mostly for internal use: */
	QuantileTreeBranch(std::unique_ptr<QuantileTreeLeaf<real>> leaf,
	                   std::function<real(real)> pdf,
	                   double atol, double rtol, tolerance_policy policy,
	                   unsigned int descent);

	static real compute_threshold(real kronrod, double atol, double rtol,
	                              tolerance_policy policy);

private:
	/* Each branch can have a maximum of one leaf. */
	std::unique_ptr<QuantileTreeLeaf<real>> leaf;
	std::unique_ptr<QuantileTreeBranch<real>> left_branch;
	std::unique_ptr<QuantileTreeBranch<real>> central_branch;
	std::unique_ptr<QuantileTreeBranch<real>> right_branch;

	/* The integral: */
	real I;

	/* Offset within the CDF: */
	real qoff;

	void check_and_split(std::function<real(real)> pdf, double atol,
	                     double rtol, tolerance_policy policy,
	                     unsigned int descent);

	/* Set the offset of this branch and all child branches. */
	void set_offsets(real qoff);
};


template<typename real>
QuantileTreeBranch<real>::QuantileTreeBranch(std::function<real(real)> pdf,
                                             real xmin, real xmax, double atol,
                                             double rtol,
                                             tolerance_policy policy)
   : QuantileTreeBranch(std::make_unique<QuantileTreeLeaf<real>>(pdf, xmin,
                                                                 xmax),
                        pdf, atol, rtol, policy, 0)
{
}


template<typename real>
real QuantileTreeBranch<real>::compute_threshold(real kronrod, double atol,
                                                 double rtol,
                                                 tolerance_policy policy)
{
	if (policy == OR){
		return 0.25 * std::max<real>(std::abs(kronrod * rtol), atol);
	}
	return 0.25 * std::min<real>(std::abs(kronrod * rtol), atol);
}


template<typename real>
QuantileTreeBranch<real>::QuantileTreeBranch(
                                  std::unique_ptr<QuantileTreeLeaf<real>> leaf_,
                                  std::function<real(real)> pdf, double atol,
                                  double rtol, tolerance_policy policy,
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
		real delta = std::abs(I - 1.0);
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


template<typename real>
void QuantileTreeBranch<real>::check_and_split(std::function<real(real)> pdf,
                                               double atol, double rtol,
                                               tolerance_policy policy,
                                               unsigned int descent)
{
	if (!leaf)
		throw std::runtime_error("check_and_split called with nullptr leaf.");

	/* Check whether the leaf is exact enough: */
	real delta_I = std::abs<real>(leaf->gauss() - leaf->kronrod());
	if (delta_I > compute_threshold(leaf->kronrod(), atol, rtol, policy)){
		/* Have to split! */
		std::array<std::unique_ptr<QuantileTreeLeaf<real>>,3>
		    leaves(leaf->split(pdf, 1.0/3.0, 2.0/3.0));

		/* Remove the leaf and add two new branches: */
		leaf.reset();
		std::unique_ptr<QuantileTreeBranch<real>>
		   new_branch = std::make_unique<QuantileTreeBranch<real>>(
		                          std::move(leaves[0]),
		                          pdf, atol, rtol, policy, descent + 1);
		left_branch.swap(new_branch);

		new_branch = std::make_unique<QuantileTreeBranch<real>>(
		                          std::move(leaves[1]),
		                          pdf, atol, rtol, policy, descent + 1);
		central_branch.swap(new_branch);

		new_branch = std::make_unique<QuantileTreeBranch<real>>(
		                          std::move(leaves[2]),
		                          pdf, atol, rtol, policy, descent + 1);
		right_branch.swap(new_branch);
	}
}


template<typename real>
void QuantileTreeBranch<real>::set_offsets(real qoff)
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


template<typename real>
std::pair<const QuantileTreeLeaf<real>&, real>
QuantileTreeBranch<real>::find(real q) const
{
	const QuantileTreeBranch<real>* branch = this;
	real qoff = 0.0;

	while (!branch->leaf){
		/* If we are in the left branch, we do not have to update
		 * qoff (the left boundaries coincide): */
		real qoff_next = qoff + branch->left_branch->I;
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

	return std::pair<const QuantileTreeLeaf<real>&,
	                 real>(*(branch->leaf), qoff);
}


template<typename real>
real QuantileTreeBranch<real>::integral() const
{
	return I;
}


template<typename real>
size_t QuantileTreeBranch<real>::leaves() const
{
	if (leaf)
		return 1;

	size_t n = 0;
	n += left_branch->leaves();
	n += central_branch->leaves();
	n += right_branch->leaves();
	return n;
}

} // namespace pdtoolbox

#endif