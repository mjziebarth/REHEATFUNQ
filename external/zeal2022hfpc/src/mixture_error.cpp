/*
 * Generator of relative measurement errors based on a mixture of two
 * normal distributions.
 * This file is part of the ziebarth_et_al_2022_heatflow python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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

#include <../include/mixture_error.hpp>

using paperheatflow::MixtureErrorGenerator;
using paperheatflow::mixture_t;

mixture_t::mixture_t(double w0, double x00, double s0, double x10, double s1)
   : w0(w0), x00(x00), s0(s0), x10(x10), s1(s1)
{
}

MixtureErrorGenerator::MixtureErrorGenerator(double w0, double x00, double s0,
                                        double x10, double s1,
                                        std::shared_ptr<std::mt19937_64> gen)
   : w0(w0), x00(x00), s0(s0), x10(x10), s1(s1), gen(gen)
{
}

MixtureErrorGenerator::MixtureErrorGenerator(const mixture_t& mix,
                                        std::shared_ptr<std::mt19937_64> gen)
   : w0(mix.w0), x00(mix.x00), s0(mix.s0), x10(mix.x10), s1(mix.s1), gen(gen)
{
}

double MixtureErrorGenerator::operator()()
{
	std::uniform_real_distribution<> uni;
	std::normal_distribution<> n0{x00, s0};
	std::normal_distribution<> n1{x10, s1};
	// Shortcut for a 100% N0 mixture model:
	if (w0 == 1.0){
		while (true){
			const double z = n0(*gen);
			if (z >= 0)
				return z;
		}
	}
	while (true){
		if (uni(*gen) <= w0){
			const double z = n0(*gen);
			if (z >= 0)
				return z;
		} else {
			const double z = n1(*gen);
			if (z >= 0)
				return z;
		}
	}
}

