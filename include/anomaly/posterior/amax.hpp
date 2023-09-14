/*
 * Heat flow anomaly analysis posterior numerics: determine maximum in `a`
 * integrand.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2021-2022 Deutsches GeoForschungsZentrum GFZ,
 *               2022-2023 Malte J. Ziebarth
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



#ifndef REHEATFUNQ_ANOMALY_POSTERIOR_AMAX_HPP
#define REHEATFUNQ_ANOMALY_POSTERIOR_AMAX_HPP

/*
 * REHEATFUNQ includes:
 */
#include <numerics/functions.hpp>
#include <anomaly/posterior/locals.hpp>
#include <anomaly/posterior/integrand.hpp>

/*
 * General includes:
 */
#include <cmath>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>

namespace reheatfunq {
namespace anomaly {
namespace posterior {

namespace rm = reheatfunq::math;

/*
 * Find the maximum of the inner integrand across all a & z
 */
template<typename real>
struct az_t {
	real a;
	real z;
	real log_integrand;
};


template<typename real>
az_t<real> log_integrand_max(const Locals<real>& L)
{
	/* Start from the middle of the interval: */
	real z = 0.5;
	real l1p_kiz_sum = 0.0;

	for (uint_fast8_t i=0; i<200; ++i){
		/* Set the new parameters:: */
		l1p_kiz_sum = 0.0;
		real k_1mkz_sum = 0.0;
		real k2_1mkz2_sum = 0.0;
		for (const real& k : L.ki){
			l1p_kiz_sum += rm::log1p(-k * z);
			/* First and second derivatives of the above by z: */
			real x = k / (1.0 - k*z);
			k_1mkz_sum -= x;
			k2_1mkz2_sum -= x*x;
		}

		const real l1p_wz = rm::log1p(-L.w * z);
		const real w_1mwz = -L.w / (1.0 - L.w*z);
		const real w2_1mwz2 = - w_1mwz * w_1mwz;


		/* New amax: */
		const real amax = std::max(log_integrand_amax(l1p_wz, l1p_kiz_sum, L),
		                           L.amin);

		/* Log of integrand:
		 * f0 =   std::lgamma(v*amax) + (amax - 1.0) * lp
		 *           - n*std::lgamma(amax) - v*amax*(ls + l1p_wz)
		 *           + (amax - 1.0) * l1p_kiz_sum
		 */

		/* Derivative of the log of the integrand by z: */
		const real f1 = - L.v * amax * w_1mwz + (amax - 1.0) * k_1mkz_sum;

		/* Second derivative of the log of the integrand by z: */
		const real f2 = - L.v * amax * w2_1mwz2 + (amax - 1.0) * k2_1mkz2_sum;

		/* Newton step: */
		real znext = std::min<real>(std::max<real>(z - f1 / f2, 0.0),
		                            1.0 - 1e-8);
		bool exit = rm::abs(znext - z) < 1e-8;
		z = znext;
		if (exit)
			break;
	}

	/* Determine amax for the final iteration: */
	l1p_kiz_sum = 0.0;
	for (const real& k : L.ki)
		l1p_kiz_sum += rm::log1p(-k * z);
	real l1p_wz = rm::log1p(-L.w * z);
	real amax = std::max(log_integrand_amax(l1p_wz, l1p_kiz_sum, L),
	                     L.amin);

	/* Log of the integrand: */
	const real f0 = rm::lgamma(L.v * amax) + (amax - 1.0) * L.lp
	                - L.n * rm::lgamma(amax) - L.v * amax * (L.ls + l1p_wz)
	                + (amax - 1.0) * l1p_kiz_sum;

	return {.a=amax, .z=z, .log_integrand=f0};
}





} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq

#endif