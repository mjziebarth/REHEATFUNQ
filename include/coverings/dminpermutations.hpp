/*
 * Algorithm for obtaining the limit distribution of minimum-distance selection
 * criterion.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2023 Malte J. Ziebarth
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

#include <vector>
#include <memory>
#include <utility>
#include <functional>

#ifndef REHEATFUNQ_COVERINGS_DMINPERMUTATIONS_HPP
#define REHEATFUNQ_COVERINGS_DMINPERMUTATIONS_HPP

namespace reheatfunq {

std::vector<std::pair<double, std::vector<size_t>>>
determine_restricted_samples(const double* xy, const size_t N,
                             const double dmin, const size_t max_samples,
                             const size_t max_iter,
                             std::shared_ptr<std::function<size_t(size_t)>>
                                sample_generator,
                             bool extra_debug_checks);


double global_PHmax(
    const double* xy,
    const double* PHmax_i,
    const size_t N,
    const double dmin
);

}

#endif