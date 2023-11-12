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

/*
 * A backend class that handles some of the template parameters:
 */
template<typename real>
class VPP_backend
{
public:
	VPP_backend(
	    const std::vector<weighted_sample_t>& weighted_samples,
	    double p, double s, double n, double v, double amin,
	    double rtol, pdf_algorithm_t algorithm,
	    size_t bli_max_splits, uint8_t bli_max_refinements
	)
	{
		if (algorithm == pdf_algorithm_t::EXPLICIT)
			posterior
			= std::make_optional<Posterior<real,pdf_algorithm_t::EXPLICIT>>(
				weighted_samples, p, s, n, v, amin, rtol,
				bli_max_splits, bli_max_refinements
			);
		else if (algorithm == pdf_algorithm_t::BARYCENTRIC_LAGRANGE)
			posterior
			= std::make_optional<Posterior<real,pdf_algorithm_t::BARYCENTRIC_LAGRANGE>>(
				weighted_samples, p, s, n, v, amin, rtol,
				bli_max_splits, bli_max_refinements
			);
		else if (algorithm == pdf_algorithm_t::ADAPTIVE_SIMPSON)
			posterior
			= std::make_optional<Posterior<real,pdf_algorithm_t::ADAPTIVE_SIMPSON>>(
				weighted_samples, p, s, n, v, amin, rtol,
				bli_max_splits, bli_max_refinements
			);
		else
			throw std::runtime_error("Could not initialize VPP_backend due to invalid "
			                         "PDF algorithm.");

	}

	double get_Qmax() const
	{
		return std::visit(
			[](auto&& p)->double
			{
				return p.get_Qmax();
			},
			*posterior
		);
	}

	void pdf(std::vector<double>& PH,
	         bool parallel=true) const
	{
		std::visit(
			[&PH, parallel](auto&& p)
			{
				p.pdf(PH, parallel);
			},
			*posterior
		);
	}

	void cdf(std::vector<double>& PH,
	         bool parallel=true) const
	{
		std::visit(
			[&PH, parallel](auto&& p)
			{
				p.cdf(PH, parallel);
			},
			*posterior
		);
	}

	void tail(std::vector<double>& PH,
	          bool parallel=true) const
	{
		std::visit(
			[&PH, parallel](auto&& p)
			{
				p.tail(PH, parallel);
			},
			*posterior
		);
	}

	void tail_quantiles(std::vector<double>& quantiles,
	                    bool parallel=true) const
	{
		std::visit(
			[&quantiles, parallel](auto&& p)
			{
				p.tail_quantiles(quantiles, parallel);
			},
			*posterior
		);
	}

	VPP_backend
	add_samples(const std::vector<weighted_sample_t>& weighted_samples) const
	{
		return std::visit(
			[&weighted_samples](auto&& p)
			{
				return VPP_backend(p.add_samples(weighted_samples));
			},
			*posterior
		);
	}

	/*
	 * Validation against the previous implementation; not for general use.
	 */
	bool validate(const std::vector<std::vector<posterior::qc_t>>& qc_set,
	              double p0, double s0, double n0, double v0,
	              double rtol) const
	{
		return std::visit(
			[=,&qc_set](auto&& p) -> bool
			{
				return p.validate(qc_set, p0, s0, n0, v0, rtol);
			},
			*posterior
		);
	}

	/*
	 * Debug output:
	 */
	void get_locals(size_t l, double& lp, double& ls, double& n, double& v,
	                double& amin, double& Qmax, std::vector<double>& ki,
	                double& h0, double& h1, double& h2, double& h3,
	                double& w, double& lh0, double& l1p_w, double& log_scale,
	                double& ymax, double& norm) const
	{
		std::visit(
			[&,l](auto&& p)
			{
				p.get_locals(l, lp, ls, n, v, amin, Qmax, ki, h0,
				             h1, h2, h3, w, lh0, l1p_w, log_scale,
							 ymax, norm);
			},
			*posterior
		);
	}

	void get_C(double a, size_t l, double& C0, double& C1, double& C2,
	           double& C3) const
	{
		std::visit(
			[&,l,a](auto&& p)
			{
				p.get_C(a, l, C0, C1, C2, C3);
			},
			*posterior
		);
	}

private:
	std::optional<std::variant<
	        Posterior<real, EXPLICIT>,
	        Posterior<real, BARYCENTRIC_LAGRANGE>,
	        Posterior<real, ADAPTIVE_SIMPSON>>>
	posterior;

	/*
	 * Construct from movable Posterior instance:
	 */
	VPP_backend(Posterior<real, EXPLICIT>&& post)
	   : posterior(std::move(post))
	{}

	VPP_backend(Posterior<real, BARYCENTRIC_LAGRANGE>&& post)
	   : posterior(std::move(post))
	{}

	VPP_backend(Posterior<real, ADAPTIVE_SIMPSON>&& post)
	   : posterior(std::move(post))
	{}


};


/*
 * The non-templated class that wraps all template parameter combinations:
 */

class VariablePrecisionPosterior
{
public:
	VariablePrecisionPosterior(
	    const std::vector<weighted_sample_t>& weighted_samples,
	    double p, double s, double n, double v, double amin,
	    double rtol, precision_t precision,
	    pdf_algorithm_t algorithm,
	    size_t bli_max_splits, uint8_t bli_max_refinements);

	double get_Qmax() const;

	std::vector<double> pdf(const std::vector<double>& PH,
	                        bool parallel=true) const;
	void pdf_inplace(std::vector<double>& PH,
	                 bool parallel=true) const;

	std::vector<double> cdf(const std::vector<double>& PH,
	                        bool parallel=true) const;
	void cdf_inplace(std::vector<double>& PH,
	                 bool parallel=true) const;

	std::vector<double> tail(const std::vector<double>& PH,
	                         bool parallel=true) const;
	void tail_inplace(std::vector<double>& PH,
	                  bool parallel=true) const;

	std::vector<double> tail_quantiles(const std::vector<double>& quantiles,
	                                   bool parallel=true) const;
	void tail_quantiles_inplace(std::vector<double>& quantiles,
	                            bool parallel=true) const;

	VariablePrecisionPosterior
	add_samples(const std::vector<weighted_sample_t>& weighted_samples) const;

	/*
	 * Validation against the previous implementation; not for general use.
	 */
	bool validate(const std::vector<std::vector<posterior::qc_t>>& qc_set,
	              double p0, double s0, double n0, double v0,
	              double rtol) const;

	/*
	 * Debug output:
	 */
	void get_locals(size_t l, double& lp, double& ls, double& n, double& v,
	                double& amin, double& Qmax, std::vector<double>& ki,
	                double& h0, double& h1, double& h2, double& h3,
	                double& w, double& lh0, double& l1p_w, double& log_scale,
	                double& ymax, double& norm) const;

	void get_C(double a, size_t l, double& C0, double& C1, double& C2,
	           double& C3) const;

private:
	const precision_t precision;
	std::optional<VPP_backend<double>> posterior_double;
	std::optional<VPP_backend<long double>> posterior_long_double;

	VariablePrecisionPosterior(const precision_t precision);

	/*
	 * The following specializations are for data types that can be toggled
	 * with Meson build switches:
	 */
	#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
	std::optional<VPP_backend<boost::multiprecision::float128>> posterior_float128;
	#endif
	#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
	std::optional<VPP_backend<boost::multiprecision::cpp_dec_float_50>> posterior_dec50;
	#endif
	#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
	std::optional<VPP_backend<boost::multiprecision::cpp_dec_float_100>> posterior_dec100;
	#endif
};

} // namespace anomaly
} // namespace reheatfunq

#endif