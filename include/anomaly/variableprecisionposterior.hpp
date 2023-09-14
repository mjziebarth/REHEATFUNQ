/*
 * Variable precision heat flow anomaly posterior.
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

#ifndef REHEATFUNQ_ANOMALY_VARIABLEPRECISIONPOSTERIOR_HPP
#define REHEATFUNQ_ANOMALY_VARIABLEPRECISIONPOSTERIOR_HPP

/*
 * General imports:
 */
#include <optional>

/*
 * REHEATFUNQ imports:
 */
#include <anomaly/posterior.hpp>
#include <ziebarth2022a.hpp>

namespace reheatfunq {
namespace anomaly {

typedef pdtoolbox::heatflow::precision_t precision_t;

class VariablePrecisionPosterior
{
public:
	VariablePrecisionPosterior(
	    const std::vector<weighted_sample_t>& weighted_samples,
	    double p, double s, double n, double v, double amin,
	    double dest_tol, precision_t precision);

	double get_Qmax() const;

	std::vector<double> pdf(const std::vector<double>& PH) const;
	void pdf_inplace(std::vector<double>& PH) const;

	std::vector<double> cdf(const std::vector<double>& PH,
	                        bool parallel=true,
	                        bool adaptive=false) const;
	void cdf_inplace(std::vector<double>& PH,
	                 bool parallel=true,
	                 bool adaptive=false) const;

	std::vector<double> tail(const std::vector<double>& PH,
	                         bool parallel=true,
	                         bool adaptive=false) const;
	void tail_inplace(std::vector<double>& PH,
	                  bool parallel=true,
	                  bool adaptive=false) const;

	std::vector<double> tail_quantiles(const std::vector<double>& quantiles,
	                                   size_t n_chebyshev=100,
	                                   bool parallel=true,
	                                   bool adaptive=false) const;
	void tail_quantiles_inplace(std::vector<double>& quantiles,
	                            size_t n_chebyshev=100, bool parallel=true,
	                            bool adaptive=false) const;

	VariablePrecisionPosterior
	add_samples(const std::vector<weighted_sample_t>& weighted_samples) const;

	/*
	 * Validation against the previous implementation; not for general use.
	 */
	bool validate(const std::vector<std::vector<posterior::qc_t>>& qc_set,
	              double p0, double s0, double n0, double v0,
	                      double dest_tol) const;


private:
	const precision_t precision;
	std::optional<Posterior<double>> posterior_double;
	std::optional<Posterior<long double>> posterior_long_double;

	VariablePrecisionPosterior(const precision_t precisoin);

	/*
	 * The following specializations are for data types that can be toggled
	 * with Meson build switches:
	 */
	#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
	std::optional<boost::multiprecision::float128> posterior_float128;
	#endif
	#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
	std::optional<boost::multiprecision::cpp_dec_float_50> posterior_dec50;
	#endif
	#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
	std::optional<boost::multiprecision::cpp_dec_float_100> posterior_dec100;
	#endif
};

} // namespace anomaly
} // namespace reheatfunq

#endif