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

template<typename real, pdf_algorithm_t pdf_algorithm>
bool
Posterior<real,pdf_algorithm>::validate(
    const std::vector<std::vector<posterior::qc_t>>& qc_set,
    double p0, double s0, double n0, double v0,
    double dest_tol
) const
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
		                                      static_cast<double>(L.amin),
		                                      dest_tol,
		                                      static_cast<double>(L.lp),
		                                      static_cast<double>(L.ls),
		                                      static_cast<double>(L.n),
		                                      static_cast<double>(L.v),
		                                      static_cast<double>(L.Qmax),
		                                      static_cast<double>(L.h[0]),
		                                      static_cast<double>(L.h[1]),
		                                      static_cast<double>(L.h[2]),
		                                      static_cast<double>(L.h[3]),
		                                      static_cast<double>(L.w),
		                                      static_cast<double>(L.lh0),
		                                      static_cast<double>(L.l1p_w))
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
namespace ra = reheatfunq::anomaly;

template bool
reheatfunq::anomaly::Posterior<long double, ra::EXPLICIT>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<long double, ra::BARYCENTRIC_LAGRANGE>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<long double, ra::ADAPTIVE_SIMPSON>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<double, ra::EXPLICIT>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<double, ra::BARYCENTRIC_LAGRANGE>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<double, ra::ADAPTIVE_SIMPSON>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;


#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::float128, ra::EXPLICIT>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::float128, ra::BARYCENTRIC_LAGRANGE>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::float128, ra::ADAPTIVE_SIMPSON>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;
#endif


#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::cpp_dec_float_50, ra::EXPLICIT>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::cpp_dec_float_50, ra::BARYCENTRIC_LAGRANGE>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::cpp_dec_float_50, ra::ADAPTIVE_SIMPSON>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;
#endif


#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::cpp_dec_float_100, ra::EXPLICIT>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::cpp_dec_float_100, ra::BARYCENTRIC_LAGRANGE>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;

template bool
reheatfunq::anomaly::Posterior<boost::multiprecision::cpp_dec_float_100, ra::ADAPTIVE_SIMPSON>::validate(
    const std::vector<std::vector<posterior::qc_t>>&,
    double, double, double, double, double
) const;
#endif
