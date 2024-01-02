/*
 * Heat flow anomaly analysis posterior numerics: convenient argument types
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2023 Malte J. Ziebarth
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

#ifndef REHEATFUNQ_ANOMALY_POSTERIOR_ARGS_HPP
#define REHEATFUNQ_ANOMALY_POSTERIOR_ARGS_HPP

namespace reheatfunq {
namespace anomaly {
namespace posterior {

template<typename real>
struct arg {
	typedef real& type;
};

template<>
struct arg<double> {
	typedef double type;
};

template<>
struct arg<long double> {
	typedef long double type;
};

template<>
struct arg<const double> {
	typedef const double type;
};

template<>
struct arg<const long double> {
	typedef const long double type;
};



}
}
}

#endif