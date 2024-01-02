/*
 * Heat flow anomaly analysis posterior numerics: integrand in `z`.
 * Source file to compile numerical constants.
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

#include <anomaly/posterior/integrand.hpp>

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
#include <boost/multiprecision/float128.hpp>
#endif

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
#include <boost/multiprecision/cpp_dec_float.hpp>
#else
    #ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
    #include <boost/multiprecision/cpp_dec_float.hpp>
    #endif
#endif

namespace reheatfunq {
namespace anomaly {
namespace posterior {

template<>
const double log_2_pi<double>::value = rm::log(2 * boost::math::constants::pi<double>());

template<>
const long double log_2_pi<long double>::value = rm::log(2 * boost::math::constants::pi<long double>());

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
template<>
const boost::multiprecision::float128
log_2_pi<boost::multiprecision::float128>::value
    = rm::log(2 * boost::math::constants::pi<boost::multiprecision::float128>());
#endif

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
template<>
const boost::multiprecision::cpp_dec_float_50
log_2_pi<boost::multiprecision::cpp_dec_float_50>::value
    = rm::log(2 * boost::math::constants::pi<boost::multiprecision::cpp_dec_float_50>());
#endif

#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
template<>
const boost::multiprecision::cpp_dec_float_100
log_2_pi<boost::multiprecision::cpp_dec_float_100>::value
    = rm::log(2 * boost::math::constants::pi<boost::multiprecision::cpp_dec_float_100>());
#endif


}
}
}
