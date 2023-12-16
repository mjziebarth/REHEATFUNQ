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

#include <anomaly/variableprecisionposterior.hpp>

using reheatfunq::anomaly::VariablePrecisionPosterior;
using reheatfunq::anomaly::pdf_algorithm_t;


VariablePrecisionPosterior::VariablePrecisionPosterior(
    const std::vector<weighted_sample_t>& weighted_samples,
    double p, double s, double n, double v, double amin,
    double rtol, precision_t precision, pdf_algorithm_t algorithm,
    size_t bli_max_splits, uint8_t bli_max_refinements)
   : precision(precision)
{
	if (precision == precision_t::WP_DOUBLE){
		posterior_double.emplace(weighted_samples, p, s, n, v, amin, rtol,
		                         algorithm, bli_max_splits, bli_max_refinements);
	} else if (precision == precision_t::WP_LONG_DOUBLE){
		posterior_long_double.emplace(weighted_samples, p, s, n, v, amin,
		                              rtol, algorithm,
		                              bli_max_splits, bli_max_refinements);
	} else if (precision == precision_t::WP_FLOAT_128){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
		posterior_float128.emplace(weighted_samples, p, s, n, v, amin,
		                           rtol, algorithm,
		                           bli_max_splits, bli_max_refinements);
		#else
		throw std::runtime_error("REHEATFUNQ is compiled without support for "
		                         "boost float128.");
		#endif
	} else if (precision == precision_t::WP_BOOST_DEC_50){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
		posterior_dec50.emplace(weighted_samples, p, s, n, v, amin,
		                        rtol, algorithm,
		                        bli_max_splits, bli_max_refinements);
		#else
		throw std::runtime_error("REHEATFUNQ is compiled without support for "
		                         "boost cpp_dec_float_50.");
		#endif
	} else if (precision == precision_t::WP_BOOST_DEC_100){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
		posterior_dec100.emplace(weighted_samples, p, s, n, v, amin,
		                         rtol, algorithm,
		                         bli_max_splits, bli_max_refinements);
		#else
		throw std::runtime_error("REHEATFUNQ is compiled without support for "
		                         "boost cpp_dec_float_100.");
		#endif
	} else {
		throw std::runtime_error("Invalid `precision` parameter.");
	}
}


double VariablePrecisionPosterior::get_Qmax() const
{
	switch (precision){
		case precision_t::WP_DOUBLE:
			return posterior_double->get_Qmax();
		case precision_t::WP_LONG_DOUBLE:
			return posterior_long_double->get_Qmax();
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			return posterior_float128->get_Qmax();
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			return posterior_dec50->get_Qmax();
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			return posterior_dec100->get_Qmax();
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}





void VariablePrecisionPosterior::pdf_inplace(std::vector<double>& PH,
                                             bool parallel) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->pdf(PH, parallel);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->pdf(PH, parallel);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->pdf(PH, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->pdf(PH, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->pdf(PH, parallel);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::pdf(const std::vector<double>& PH,
                                bool parallel) const
{
	std::vector<double> result = PH;
	pdf_inplace(result, parallel);
	return result;
}


void VariablePrecisionPosterior::cdf_inplace(std::vector<double>& PH,
                                             bool parallel) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->cdf(PH, parallel);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->cdf(PH, parallel);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->cdf(PH, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->cdf(PH, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->cdf(PH, parallel);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::cdf(const std::vector<double>& PH,
                                bool parallel) const
{
	std::vector<double> result = PH;
	cdf_inplace(result, parallel);
	return result;
}


void VariablePrecisionPosterior::tail_inplace(std::vector<double>& PH,
                                              bool parallel) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->tail(PH, parallel);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->tail(PH, parallel);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->tail(PH, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->tail(PH, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->tail(PH, parallel);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::tail(const std::vector<double>& PH,
                                 bool parallel) const
{
	std::vector<double> result = PH;
	tail_inplace(result, parallel);
	return result;
}


void VariablePrecisionPosterior::tail_quantiles_inplace(
           std::vector<double>& quantiles,
           bool parallel) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->tail_quantiles(quantiles, parallel);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->tail_quantiles(quantiles, parallel);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->tail_quantiles(quantiles, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->tail_quantiles(quantiles, parallel);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->tail_quantiles(quantiles, parallel);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::tail_quantiles(const std::vector<double>& quantiles,
	                                       bool parallel) const
{
	std::vector<double> result = quantiles;
	tail_quantiles_inplace(result, parallel);
	return result;
}


VariablePrecisionPosterior::VariablePrecisionPosterior(const precision_t pr)
   : precision(pr)
{}


VariablePrecisionPosterior
VariablePrecisionPosterior::add_samples(
     const std::vector<weighted_sample_t>& weighted_samples
) const
{
	/*
	 * Return values:
	 */
	VariablePrecisionPosterior vpp(precision);
	/*
	 * The relevant optionals have been initialized on start.
	 */

	switch (precision){
		case precision_t::WP_DOUBLE:
			vpp.posterior_double.emplace(
			    posterior_double->add_samples(weighted_samples)
			);
			return vpp;
		case precision_t::WP_LONG_DOUBLE:
			vpp.posterior_long_double.emplace(
			    posterior_long_double->add_samples(weighted_samples)
			);
			return vpp;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			vpp.posterior_float128.emplace(
			    posterior_float128->add_samples(weighted_samples)
			);
			return vpp;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			vpp.posterior_dec50.emplace(
			    posterior_dec50->add_samples(weighted_samples)
			);
			return vpp;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			vpp.posterior_dec100.emplace(
			    posterior_dec100->add_samples(weighted_samples)
			);
			return vpp;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}





bool
VariablePrecisionPosterior::validate(
     const std::vector<std::vector<posterior::qc_t>>& qc_set,
     double p0, double s0, double n0, double v0, double rtol
) const
{
	switch (precision){
		case precision_t::WP_DOUBLE:
			return posterior_double->validate(qc_set, p0, s0, n0, v0, rtol);
		case precision_t::WP_LONG_DOUBLE:
			return posterior_long_double->validate(qc_set, p0, s0, n0, v0,
			                                       rtol);
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			return posterior_float128->validate(qc_set, p0, s0, n0, v0,
			                                    rtol);
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			return posterior_dec50->validate(qc_set, p0, s0, n0, v0, rtol);
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			return posterior_dec100->validate(qc_set, p0, s0, n0, v0, rtol);
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}


void VariablePrecisionPosterior::get_locals(size_t l, double& lp, double& ls,
		double& n, double& v, double& amin, double& Qmax, std::vector<double>& ki,
	    double& h0, double& h1, double& h2, double& h3, double& w, double& lh0,
		double& l1p_w, double& log_scale, double& ymax, double& norm) const
{
	switch (precision){
		case precision_t::WP_DOUBLE:
			return posterior_double->get_locals(l, lp, ls, n, v, amin, Qmax, ki,
			                                    h0, h1, h2, h3, w, lh0, l1p_w,
			                                         log_scale, ymax, norm);
		case precision_t::WP_LONG_DOUBLE:
			return posterior_long_double->get_locals(l, lp, ls, n, v, amin, Qmax,
			                                         ki, h0, h1, h2, h3, w, lh0,
			                                         l1p_w, log_scale, ymax, norm);
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			return posterior_float128->get_locals(l, lp, ls, n, v, amin, Qmax, ki,
			                                      h0, h1, h2, h3, w, lh0, l1p_w,
			                                      log_scale, ymax, norm);
			#else
			throw std::runtime_error("Code not compiled with float128 support.");
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			return posterior_dec50->get_locals(l, lp, ls, n, v, amin, Qmax, ki,
			                                   h0, h1, h2, h3, w, lh0, l1p_w,
			                                   log_scale, ymax, norm);
			#else
			throw std::runtime_error("Code not compiled with boost dec50 support.");
			#endif
			break;
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			return posterior_dec100->get_locals(l, lp, ls, n, v, amin, Qmax, ki,
			                                    h0, h1, h2, h3, w, lh0, l1p_w,
			                                    log_scale, ymax, norm);
			#else
			throw std::runtime_error("Code not compiled with boost dec100 support.");
			#endif
			break;
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}


void VariablePrecisionPosterior::get_C(double a, size_t l, double& C0,
     double& C1, double& C2, double& C3
) const
{
	switch (precision){
		case precision_t::WP_DOUBLE:
			return posterior_double->get_C(a, l, C0, C1, C2, C3);
		case precision_t::WP_LONG_DOUBLE:
			return posterior_long_double->get_C(a, l, C0, C1, C2, C3);
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			return posterior_float128->get_C(a, l, C0, C1, C2, C3);
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			return posterior_dec50->get_C(a, l, C0, C1, C2, C3);
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			return posterior_dec100->get_C(a, l, C0, C1, C2, C3);
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}




void
VariablePrecisionPosterior::get_log_pdf_bli_samples(
    std::vector<std::vector<std::pair<double,double>>>& samples
) const
{
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->get_log_pdf_bli_samples(samples);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->get_log_pdf_bli_samples(samples);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->get_log_pdf_bli_samples(samples);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->get_log_pdf_bli_samples(samples);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->get_log_pdf_bli_samples(samples);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}
