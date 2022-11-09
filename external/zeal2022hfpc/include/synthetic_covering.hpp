/*
 * Code to generate a synthetic covering.
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

#include <types.hpp>

#ifndef PAPER_HEATFLOW_SYNTHETIC_COVERING_HPP
#define PAPER_HEATFLOW_SYNTHETIC_COVERING_HPP

namespace paperheatflow {

/*
 * Mixture of 2 normal distributions.
 */
std::vector<covering_t>
generate_synthetic_heat_flow_coverings_mixture(
     const std::vector<sample_params_t>& sample_params, size_t N, double hf_max,
     double w0, double x00, double s0, double x10, double s1,
     size_t seed, unsigned short nthread);

/*
 * Mixture of 3 normal distributions.
 */
std::vector<covering_t>
generate_synthetic_heat_flow_coverings_mixture(
     const std::vector<std::vector<sample_params_t>>& sample_params,
     double hf_max, double w0, double x00, double s0,
     double w1, double x10, double s1, double x20, double s2,
     size_t seed, unsigned short nthread);

/*
 * Generate random numbers:
 */
std::vector<double>
mixture_normal_3(size_t N, double w0, double x00, double s0, double w1,
                 double x10, double s1, double x20, double s2, size_t seed);


}

#endif