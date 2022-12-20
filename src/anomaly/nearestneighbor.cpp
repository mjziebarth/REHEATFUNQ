/*
 * Nearest neighbor interpolated heat flow anomaly.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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

#include <anomaly/nearestneighbor.hpp>

#include <cmath>
#include <boost/geometry.hpp>

using namespace reheatfunq;
using boost::geometry::index::nearest;


NearestNeighborAnomaly::tree_t
NearestNeighborAnomaly::assemble_tree(const double* xyc, size_t N)
{
	tree_t tree;
	if (N == 0)
		return tree;

	for (size_t i=0; i<N; ++i){
		double x = *xyc;
		++xyc;
		double y = *xyc;
		++xyc;
		++xyc;
		tree.insert(std::make_pair(xy_t(x,y), i));
	}

	return tree;
}

static std::vector<double> extract_c(const double* xyc, size_t N)
{
	std::vector<double> c(N);
	if (N == 0)
		return c;

	for (size_t i=0; i<N; ++i){
		c[i] = *(xyc + 3*i + 2);
	}

	return c;
}


NearestNeighborAnomaly::NearestNeighborAnomaly(
    const double* xyc,
    size_t N
) : tree(assemble_tree(xyc, N)), c(extract_c(xyc, N))
{
}


double NearestNeighborAnomaly::c_i(double x, double y, double P_H) const
{
	return _c_i(x, y, P_H);
}


double NearestNeighborAnomaly::_c_i(double x, double y, double P_H) const
{
	/* Back inserter for a single element: */
	struct closest_t {
		typedef std::pair<xy_t,size_t> value_type;

		size_t id = 0;

		void push_back(const value_type& s) {
			id = s.second;
		};
	};

	/* Find the closest fault trace segment: */
	const xy_t p(x,y);
	closest_t closest;
	tree.query(nearest(p, 1), std::back_inserter(closest));

	/* Return the value: */
	return c.at(closest.id);
}


void NearestNeighborAnomaly::batch_c_i_ptr(size_t N, const double* xy,
                                           double* c_i, double P_H) const
{
	for (size_t i=0; i<N; ++i){
		const double x = *xy;
		++xy;
		const double y = *xy;
		++xy;
		*c_i = _c_i(x, y, P_H);
		++c_i;
	}
}
