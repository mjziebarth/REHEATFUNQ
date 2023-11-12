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
#include <algorithm>
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
	struct state_t {
		real l1p_kiz_sum;
		real k_1mkz_sum;
		real k2_1mkz2_sum;
		real l1p_wz;
		real w_1mwz;
		real w2_1mwz2;
		real amax;
		real f0;

		void compute_f0(const Locals<real>& L) {
			f0 = rm::lgamma(L.v * amax) + (amax - 1.0) * L.lp
			     - L.n * rm::lgamma(amax) - L.v * amax * (L.ls + l1p_wz)
			     + (amax - 1.0) * l1p_kiz_sum;
		};

		void compute(real z, const Locals<real>& L){
			/* Set the new parameters:: */
			l1p_kiz_sum = 0.0;
			k_1mkz_sum = 0.0;
			k2_1mkz2_sum = 0.0;
			for (const real& k : L.ki){
				l1p_kiz_sum += rm::log1p(-k * z);
				/* First and second derivatives of the above by z: */
				real x = k / (1.0 - k*z);
				k_1mkz_sum -= x;
				k2_1mkz2_sum -= x*x;
			}

			l1p_wz = rm::log1p(-L.w * z);
			w_1mwz = -L.w / (1.0 - L.w*z);
			w2_1mwz2 = - w_1mwz * w_1mwz;


			/* New amax: */
			amax = std::max(log_integrand_amax(l1p_wz, l1p_kiz_sum, L),
							L.amin);

		};
	};
	state_t state;

	/* Find a starting value. Identify a maximum from a regular grid
	 * in z:
	 */
	uint_fast8_t imax=0;
	real f0max = -std::numeric_limits<real>::infinity();
	for (uint_fast8_t i=1; i<20; ++i){
		state.compute(i / 20.0, L);
		state.compute_f0(L);
		if (state.f0 > f0max){
			imax = i;
			f0max = state.f0;
		}
	}

	/* Reduce our maximum search to the interval [i-1, i+1]: */
	real zl, zr;
	if (imax == 0){
		zl = 0;
		zr = 1/20.0;
	} else if (imax == 19){
		zl = 18/20.0;
		zr = 1.0;
	} else {
		zl = (imax - 1) / 20.0;
		zr = (imax + 1) / 20.0;
	}

	/* Start from the middle of the interval: */
	real z = (zl + zr) / 2;

	for (uint_fast8_t i=0; i<200; ++i){
		state.compute(z, L);

		/* Derivative of the log of the integrand by z: */
		const real f1 = - L.v * state.amax * state.w_1mwz + (state.amax - 1.0) * state.k_1mkz_sum;

		/* Second derivative of the log of the integrand by z: */
		const real f2 = - L.v * state.amax * state.w2_1mwz2 + (state.amax - 1.0) * state.k2_1mkz2_sum;

		/* Newton step: */
		real dz = std::min<real>(
			std::max<real>(
				- f1 / f2,
				0.99 * (zl - z)
			),
			0.99 * (zr - z)
		);
		real znext = z + dz;
		bool exit = rm::abs(znext - z) < 1e-8;
		z = znext;
		if (exit)
			break;
	}

	state.compute(z, L);
	state.compute_f0(L);

	return {.a=state.amax, .z=z, .log_integrand=state.f0};
}





} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq

#endif