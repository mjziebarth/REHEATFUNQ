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
#include <numerics/limits.hpp>
#include <anomaly/posterior/locals.hpp>
#include <anomaly/posterior/integrand.hpp>

/*
 * General includes:
 */
#include <cmath>
#include <algorithm>
#ifndef BOOST_ENABLE_ASSERT_HANDLER
#define BOOST_ENABLE_ASSERT_HANDLER // Make sure the asserts do not abort
#endif
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/tools/minima.hpp>


namespace reheatfunq {
namespace anomaly {
namespace posterior {

namespace rm = reheatfunq::math;
namespace rn = reheatfunq::numerics;

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
		real l1p_wz;
		real amax;
		real f0;

		void compute_f0(const Locals<real>& L) {
			real C = L.lp - L.v * (L.ls + l1p_wz) + l1p_kiz_sum;
			f0 = loggamma_v_a__minus__n_loggamma_a__plus__C_a<real>(amax, L.n, L.v, C, L.lv)
			      - (L.lp + l1p_kiz_sum);
		};

		void compute(real z, const Locals<real>& L){
			/* Set the new parameters:: */
			l1p_kiz_sum = 0.0;
			for (const real& k : L.ki){
				l1p_kiz_sum += rm::log1p(-k * z);
			}

			l1p_wz = rm::log1p(-L.w * z);

			/* New amax: */
			amax = std::max(log_integrand_amax(l1p_wz, l1p_kiz_sum, L),
							L.amin);

		};
	};
	state_t state;


	auto cost = [&L](real z) -> real {
		if (z == 1.0)
			return std::numeric_limits<real>::infinity();

		/* Compute the state at z: */
		state_t state;
		state.compute(z, L);
		state.compute_f0(L);
		return -state.f0;
	};

	std::uintmax_t max_iter = 200;
	std::pair<real,real> res
	   = boost::math::tools::brent_find_minima(
	          cost,
	          static_cast<real>(0.0),
	          static_cast<real>(1.0),
	          1000000,
	          max_iter
	);

	if (rm::isnan(res.first))
		throw std::runtime_error("NaN z in log_integrand_max.");

	/* Evaluate for the z obtained in the Newton-Raphson iteration: */
	state.compute(res.first, L);
	real amax_z = state.amax;

	if (rm::isnan(amax_z))
		throw std::runtime_error("NaN a in log_integrand_max.");

	return {.a=amax_z, .z=res.first, .log_integrand=-res.second};
}







} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq

#endif