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

#include <random>
#include <memory>

#ifndef PAPER_HEATFLOW_MIXTURE_ERROR_HPP
#define PAPER_HEATFLOW_MIXTURE_ERROR_HPP

namespace paperheatflow {

struct mixture_t {
	double w0;
	double x00;
	double s0;
	double x10;
	double s1;
	
	mixture_t(double w0, double x00, double s0, double x10, double s1);
};

class MixtureErrorGenerator {
public:
	typedef mixture_t params_t;

	MixtureErrorGenerator(double w0, double x00, double s0, double x10,
	                      double s1, std::shared_ptr<std::mt19937_64> gen);

	MixtureErrorGenerator(const mixture_t& mix,
	                      std::shared_ptr<std::mt19937_64> gen);

	double operator()();

private:
	const double w0, x00, s0, x10, s1;
	std::shared_ptr<std::mt19937_64> gen;
};

}

#endif