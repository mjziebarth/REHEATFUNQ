/*
 * Specifications of some numeric limits.
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

#ifndef REHEATFUNQ_NUMERICS_LIMITS_HPP
#define REHEATFUNQ_NUMERICS_LIMITS_HPP

#include <limits>

namespace reheatfunq {
namespace numerics {

template<typename real>
class numeric_limits {
public:
    static constexpr real epsilon() noexcept {
        return std::numeric_limits<real>::epsilon();
    }
};

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
template<>
class numeric_limits<boost::multiprecision::float128>
{
public:
    static constexpr long double epsilon() noexcept {
        return 1.9259299443872358531e-34;
    }
};
#endif

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
template<>
class numeric_limits<boost::multiprecision::cpp_dec_float_50>
{
public:
    static constexpr long double epsilon() noexcept {
        return 1e-49;
    }
};
#endif

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
template<>
class numeric_limits<boost::multiprecision::cpp_dec_float_100>
{
public:
    static constexpr long double epsilon() noexcept {
        return 1e-99;
    }
};
#endif


}
}

#endif