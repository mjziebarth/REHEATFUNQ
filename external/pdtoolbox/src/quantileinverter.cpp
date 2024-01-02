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

#include <../include/quantiletree/leaf.hpp>
#include <string>
#include <iostream>

using pdtoolbox::QuantileTreeLeaf;

/*
 * Gauss-Kronrod constants:
 *
 * double:
 */
#ifndef BOOST_ENABLE_ASSERT_HANDLER
#define BOOST_ENABLE_ASSERT_HANDLER // Make sure the asserts do not abort
#endif
#include <boost/math/quadrature/gauss_kronrod.hpp>
using boost::math::quadrature::detail::gauss_kronrod_detail;
using boost::math::quadrature::detail::gauss_detail;

template<>
const std::array<double,8> QuantileTreeLeaf<double>::gk75_abscissa
    = gauss_kronrod_detail<double,15,1>::abscissa();

template<>
const std::array<double,8> QuantileTreeLeaf<double>::k7_weights
    = gauss_kronrod_detail<double,15,1>::weights();

template<>
const std::array<double,4> QuantileTreeLeaf<double>::g5_weights
    = gauss_detail<double,7,1>::weights();

template<>
const std::array<double,16> QuantileTreeLeaf<double>::k7_trapezoid_mass
    = QuantileTreeLeaf<double>::compute_trapezoid_mass();

template<>
const std::array<double,17> QuantileTreeLeaf<double>::scaled_k7_lr_abscissa
   = QuantileTreeLeaf<double>::compute_scaled_k7_lr_abscissa();


/*
 * long double
 */
template<>
const std::array<long double,8> QuantileTreeLeaf<long double>::gk75_abscissa
    = gauss_kronrod_detail<long double,15,1>::abscissa();

template<>
const std::array<long double,8> QuantileTreeLeaf<long double>::k7_weights
    = gauss_kronrod_detail<long double,15,1>::weights();

template<>
const std::array<long double,4> QuantileTreeLeaf<long double>::g5_weights
    = gauss_detail<long double,7,1>::weights();

template<>
const std::array<long double,16>
QuantileTreeLeaf<long double>::k7_trapezoid_mass
    = QuantileTreeLeaf<long double>::compute_trapezoid_mass();

template<>
const std::array<long double,17>
QuantileTreeLeaf<long double>::scaled_k7_lr_abscissa
   = QuantileTreeLeaf<long double>::compute_scaled_k7_lr_abscissa();
