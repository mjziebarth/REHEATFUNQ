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

#include <anomaly.hpp>

#include <utility>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/index/rtree.hpp>


#ifndef REHEATFUNQ_ANOMALY_LS1980_HPP
#define REHEATFUNQ_ANOMALY_LS1980_HPP

namespace reheatfunq {

class LachenbruchSass1980Anomaly : public HeatFlowAnomaly {
public:
	LachenbruchSass1980Anomaly(const double* xy, size_t N, double d);

	virtual ~LachenbruchSass1980Anomaly() = default;

	virtual double c_i(double x, double y, double P_H=1.0) const;

	virtual void batch_c_i_ptr(size_t N, const double* xy, double* c_i,
	                           double P_H=1.0) const;

private:
	/*
	 * Boost typedefs
	 */
	typedef boost::geometry::model::d2::point_xy<double> xy_t;
	typedef boost::geometry::model::segment<xy_t> seg_t;
	typedef boost::geometry::index::rstar<16> build_t;
	typedef boost::geometry::index::rtree<std::pair<seg_t,size_t>,
	                                      build_t> tree_t;


	/* Fault trace depth: */
	const double d;
	/* Fault trace length: */
	const double L;

	/* Spatial index: */
	const tree_t tree;

	static tree_t assemble_tree(const double* xy, size_t N);

	double _c_i(double x, double y, double P_H) const;
};

}

#endif