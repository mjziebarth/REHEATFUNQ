/*
 * Posterior of the volume.
 * NOTE: This development file is not currently in use.
 *
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

#include <cstddef>
#include <vector>

#ifndef PAPER_HEATFLOW_VOLUME_POSTERIOR_HPP
#define PAPER_HEATFLOW_VOLUME_POSTERIOR_HPP

namespace paperheatflow {

struct heat_t {
	double q_mW_m2;
	double c_1_m2;
};

void compute_posterior_volume_parameters(const double* efficiency, size_t Neff,
         const double* power_GW, size_t Npow,
         const std::vector<std::vector<double>>& xis,
         const std::vector<double>& weights,
         const std::vector<std::vector<heat_t>>& heat_flow_data,
         double prior_p, double prior_s, double prior_n, double prior_v,
         bool renorm_P_H_dimension, bool efficiency_similar,
         double max_efficiency_saf, size_t n_ph, size_t n_interp,
         double* posterior_out, double *mass_out);


}

#endif