/*
 * Gamma distribution maximum likelihood estimate.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2021-2022 Deutsches GeoForschungsZentrum GFZ,
 *               2022 Malte J. Ziebarth
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

#include <gamma.hpp>
#include <cmath>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

using boost::math::digamma, boost::math::trigamma;

namespace pdtoolbox {

gamma_mle_t compute_gamma_mle(const double mean_x, const double mean_logx,
                              const double amin)
{
	/*
	 * Initial guess:
	 * From: Minka, Thomas P. (2002): "Estimating a Gamma distribution."
	 */
	const double s = std::log(mean_x) - mean_logx;
	double a = (3.0 - s + std::sqrt((s - 3.0)*(s - 3.0) + 24.0*s)) / (12.0*s);

	/*
	 * Newton-Raphson iteration:
	 */
	double a1;
	a1 = a - (std::log(a) - digamma(a) - s) \
	                / (1.0/a - trigamma(a));
	size_t i = 0;
	while (std::fabs(a1 - a) > 1e-13 && i < 20){
		a = a1;
		a1 = a - (std::log(a) - digamma(a) - s) \
		          / (1.0/a - trigamma(a));
		a1 = std::max(a1, amin);
		++i;
	}

	/*
	 * Set and return the final results:
	 */
	gamma_mle_t res;
	res.a = a1;
	res.b = res.a / mean_x;

	return res;
}



}
