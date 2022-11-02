/*
 * Code to represent heat flow anomalies.
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

#include <cstddef>

#ifndef REHEATFUNQ_ANOMALY_HPP
#define REHEATFUNQ_ANOMALY_HPP

namespace reheatfunq {

class HeatFlowAnomaly {
public:
	virtual ~HeatFlowAnomaly() = default;

	virtual double c_i(double x, double y, double P_H = 1.0) const;

	virtual void batch_c_i_ptr(size_t N, const double* xy, double* c_i,
	                           double P_H=1.0) const;
};


}

#endif