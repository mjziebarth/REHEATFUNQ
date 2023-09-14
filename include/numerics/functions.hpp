/*
 * Math functions.
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

#ifndef REHEATFUNQ_NUMERICS_MATH_HPP
#define REHEATFUNQ_NUMERICS_MATH_HPP

#include <cmath>
#define BOOST_ENABLE_ASSERT_HANDLER // Make sure the asserts do not abort
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace reheatfunq {
namespace math {

using std::exp;
using boost::multiprecision::exp;

using std::log;
using boost::multiprecision::log;

using std::log1p;
using boost::multiprecision::log1p;

using std::lgamma;
using boost::multiprecision::lgamma;

using std::abs;
using boost::multiprecision::abs;

using std::isinf;
using boost::multiprecision::isinf;

using std::isnan;
using boost::multiprecision::isnan;

using boost::math::digamma;

using boost::math::trigamma;

} // namespace math
} // namespace reheatfunq

#endif