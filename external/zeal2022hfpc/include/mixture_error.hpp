/*
 * Generator of relative measurement errors based on a mixture of two
 * normal distributions.
 * This file is part of the ziebarth_et_al_2022_heatflow python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
 *                    Malte J. Ziebarth
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

#include <array>
#include <random>
#include <memory>
#include <type_traits>
#include <stdexcept>

#ifndef PAPER_HEATFLOW_MIXTURE_ERROR_HPP
#define PAPER_HEATFLOW_MIXTURE_ERROR_HPP

namespace paperheatflow {

template<uint8_t n, typename std::enable_if_t<(n > 1),int>* = nullptr>
struct mixture_t {
	std::array<double,n-1> w_cumul;
	std::array<double,n> x0;
	std::array<double,n> s;

	mixture_t(std::array<double, n-1>&& w_,
	          std::array<double, n>&& x0_,
	          std::array<double, n>&& s_)
	   : w_cumul(std::move(w_)), x0(std::move(x0_)), s(std::move(s_))
	{
		/* Cumulative weight sum: */
		for (uint8_t i=1; i<n-1; ++i){
			w_cumul[i] += w_cumul[i-1];
		}
		if (w_cumul[n-2] > 1.0)
			throw std::runtime_error("Sum of weights > 1!");
	};
};

template<uint8_t n, typename std::enable_if_t<(n > 0),int>* = nullptr>
class MixtureErrorGenerator {
public:
	typedef mixture_t<n> params_t;

	MixtureErrorGenerator(const params_t& mix,
	                      std::shared_ptr<std::mt19937_64> gen)
	   : p(mix), gen(gen)
	{
	};

	double operator()()
	{
		std::uniform_real_distribution<> uni;
		// Shortcut for a 100% N0 mixture model:
		if (p.w_cumul[0] == 1.0){
			while (true){
				const double z
				   = std::normal_distribution<>(p.x0[0], p.s[0])(*gen);
				if (z >= 0)
					return z;
			}
		}

		while (true){
			/* Select the mixture to choose: */
			const double w = uni(*gen);

			for (uint_fast8_t i=0; i<n; ++i){
				/* Check if the generated weight is within the
				 * current mixture component's range: */
				if (i+1 == n || w <= p.w_cumul[i]){
					const double z
					   = std::normal_distribution<>(p.x0[i], p.s[i])(*gen);
					if (z >= 0)
						return z;
					break;
				}
			}
		}
	}

private:
	const params_t p;
	std::shared_ptr<std::mt19937_64> gen;
};

}

#endif