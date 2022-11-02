/*
 * Heat flow anomaly by Lachenbruch & Sass (1980).
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

#include <anomaly/ls1980.hpp>
#include <cmath>
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/distance.hpp>

using namespace reheatfunq;
using boost::geometry::index::nearest;

/*
 * Compute the fault length assuming that `xy` is an array of
 * coordinates like (x(0), y(0), x(1), y(1), ..., x(N-1), y(N-1))
 */
static double fault_length(const double* xy, size_t N)
{
	if (N <= 1)
		return 0.0;
	double length = 0.0;
	double x = *xy;
	++xy;
	double y = *xy;
	++xy;
	for (size_t i=1; i<N; ++i){
		/* Get the new coordinates: */
		double xi = *xy;
		++xy;
		double yi = *xy;
		++xy;
		/* Compute the distance: */
		double dx = xi - x;
		double dy = yi - y;
		length += std::sqrt(dx*dx + dy*dy);
		x = xi;
		y = yi;
	}
	return length;
}


LachenbruchSass1980Anomaly::tree_t
LachenbruchSass1980Anomaly::assemble_tree(const double* xy, size_t N)
{
	tree_t tree;
	if (N <= 1)
		return tree;

	/* Last point: */
	double x = *xy;
	++xy;
	double y = *xy;
	++xy;
	xy_t last(x,y);

	for (size_t i=1; i<N; ++i){
		x = *xy;
		++xy;
		y = *xy;
		++xy;
		xy_t next(x,y);
		tree.insert(std::make_pair(seg_t(last, next), i));
		last = next;
	}

	return tree;
}


LachenbruchSass1980Anomaly::LachenbruchSass1980Anomaly(
    const double* xy,
    size_t N,
    double d
) : d(d), L(fault_length(xy, N)), tree(assemble_tree(xy,N))
{
}


double LachenbruchSass1980Anomaly::c_i(double x, double y, double P_H) const
{
	return _c_i(x, y, P_H);
}


double LachenbruchSass1980Anomaly::_c_i(double x, double y, double P_H) const
{
	/* Back inserter for a single element: */
	struct closest_t {
		typedef std::pair<seg_t,size_t> value_type;

		seg_t val;

		void push_back(const value_type& s) {
			val = s.first;
		};
	};

	/* Find the closest fault trace segment: */
	const xy_t p(x,y);
	closest_t closest;
	tree.query(nearest(p, 1), std::back_inserter(closest));

	/* Compute the distance to it: */
	const double dist = boost::geometry::distance(p, closest.val);

	/* Evaluate the anomaly: */
	const double Qstar = 2.0 * P_H / (d * L);
	return Qstar / M_PI * (1.0 - dist / d * std::atan(d / dist));
}


void LachenbruchSass1980Anomaly::batch_c_i_ptr(size_t N, const double* xy,
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
