/*
 * Heat flow anomaly analysis posterior numerics.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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
#include <anomaly/posterior.hpp>
#include <ziebarth2022a.hpp>


namespace reheatfunq {
namespace anomaly {

template<typename real>
bool
Posterior<real>::validate(const std::vector<std::vector<posterior::qc_t>>&
                                qc_set,
                          double p0, double s0, double n0, double v0,
                          double dest_tol) const
{
	if (qc_set.size() != locals.size())
		throw std::runtime_error("Size of provided set to test against is not "
		                         "correct.");
	size_t i=0;
	for (const posterior::Locals<real>& L : locals)
	{
		/* Compute the reference parameters: */
		std::vector<double> qi(qc_set[i].size());
		std::vector<double> ci(qc_set[i].size());
		for (size_t j=0; j<qc_set[i].size(); ++j){
			qi[j] = qc_set[i][j].q;
			ci[j] = qc_set[i][j].c;
		}

		if (pdtoolbox::heatflow::check_locals(qi.data(), ci.data(), qi.size(),
		                                      p0, s0, n0, v0,
		                                      L.amin, dest_tol, L.lp, L.ls,
		                                      L.n, L.v, L.Qmax, L.h[0], L.h[1],
		                                      L.h[2], L.h[3], L.w, L.lh0,
		                                      L.l1p_w)
		    != 0)
		{
			return false;
		}

		++i;
	}

	return true;
}

} // namespace reheatfunq
} // namespace anomaly

/*
 * Explicitly specialize:
 */
template bool
reheatfunq::anomaly::Posterior<long double>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<double>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;
