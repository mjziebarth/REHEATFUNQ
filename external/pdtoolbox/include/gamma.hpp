/*
 * Gamma distribution maximum likelihood estimate.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Malte J. Ziebarth
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

#ifndef REHEATFUNQ_PDTB_GAMMA_HPP
#define REHEATFUNQ_PDTB_GAMMA_HPP

namespace pdtoolbox {

struct gamma_mle_t {
	double a;
	double b;
};

gamma_mle_t compute_gamma_mle(const double mean_x, const double mean_logx,
                              const double amin);

}

#endif