/*
 * Constexpr implementations of mathematical functions.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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

#include <cmath>

namespace pdtoolbox {

/*
 * Square root using the Babylon algorithm.
 * Adds some 'static asserts' by raising recursion depth errors.
 */
constexpr double const_sqrt_iter(const double S, const double d)
{
	/* A very hacky static assert, rasing a recursion depth error: */
	if (S < 0)
		return const_sqrt_iter(S,d);

	double Snext = (S + d/S) / 2;
	if (Snext == S)
		return S;
	return const_sqrt_iter(Snext, d);
}

constexpr double cnst_sqrt(const double d)
{
	/* Babylon: */
	double S = const_sqrt_iter(d,d);

	/* "static_assert" */
	if (std::abs(S*S - d) > 1e-14*d)
		return cnst_sqrt(d);

	return S;
}

}