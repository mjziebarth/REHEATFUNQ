/*
 * Resilience and performance analysis.
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
#include <optional>
#include <vector>
#include <array>

#ifndef REHEATFUNQ_RESILIENCE_RESILIENCE_HPP
#define REHEATFUNQ_RESILIENCE_RESILIENCE_HPP


namespace reheatfunq {
namespace resilience {

struct quantiles_t {
	double proper;
	double improper;
};

std::optional<std::vector<quantiles_t>>
test_performance_1q(size_t N, size_t M, double P_MW, double K, double T,
                    double quantile, double PRIOR_P, double PRIOR_S,
                    double PRIOR_N, double PRIOR_V, double amin, bool verbose,
                    bool show_failures, size_t seed,
                    unsigned short nthread, double tolerance);

std::optional<std::vector<quantiles_t>>
test_performance_41q(size_t N, size_t M, double P_MW, double K, double T,
                     const std::array<double,41>& quantile, double PRIOR_P,
                     double PRIOR_S, double PRIOR_N, double PRIOR_V,
                     double amin, bool verbose, bool show_failures, size_t seed,
                     unsigned short nthread, double tolerance);

std::optional<std::vector<quantiles_t>>
test_performance_mixture_4q(size_t N, size_t M, double P_MW, double x0,
                    double s0, double a0, double x1, double s1, double a1,
                    const std::array<double,4>& quantiles,
                    double PRIOR_P, double PRIOR_S, double PRIOR_N,
                    double PRIOR_V, double amin, bool verbose,
                    bool show_failures, size_t seed, unsigned short nthread,
                    double tolerance);

std::optional<std::vector<quantiles_t>>
test_performance_mixture_41q(size_t N, size_t M, double P_MW, double x0,
                    double s0, double a0, double x1, double s1, double a1,
                    const std::array<double,41>& quantiles,
                    double PRIOR_P, double PRIOR_S, double PRIOR_N,
                    double PRIOR_V, double amin, bool verbose,
                    bool show_failures, size_t seed, unsigned short nthread,
                    double tolerance);


}
}

#endif