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


VariablePrecisionPosterior::VariablePrecisionPosterior(
    const std::vector<weighted_sample_t>& weighted_samples,
    double p, double s, double n, double v, double amin,
    double dest_tol, precision_t precision)
   : precision(precision)
{
	if (precision == precision_t::WP_DOUBLE){
		posterior_double.emplace(weighted_samples, p, s, n, v, amin, dest_tol);
	} else if (precision == precision_t::WP_LONG_DOUBLE){
		posterior_long_double.emplace(weighted_samples, p, s, n, v, amin,
		                              dest_tol);
	} else if (precision == precision_t::WP_FLOAT_128){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
		posterior_float128.emplace(weighted_samples, p, s, n, v, amin,
		                           dest_tol);
		#else
		throw std::runtime_error("REHEATFUNQ is compiled without support for "
		                         "boost float128.");
		#endif
	} else if (precision == precision_t::WP_BOOST_DEC_50){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
		posterior_dec50.emplace(weighted_samples, p, s, n, v, amin,
		                        dest_tol);
		#else
		throw std::runtime_error("REHEATFUNQ is compiled without support for "
		                         "boost cpp_dec_float_50.");
		#endif
	} else if (precision == precision_t::WP_BOOST_DEC_100){
		#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
		posterior_dec100.emplace(weighted_samples, p, s, n, v, amin,
		                         dest_tol);
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





void VariablePrecisionPosterior::pdf_inplace(std::vector<double>& PH) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->pdf(PH);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->pdf(PH);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->pdf(PH);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->pdf(PH);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->pdf(PH);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::pdf(const std::vector<double>& PH) const
{
	std::vector<double> result = PH;
	pdf_inplace(result);
	return result;
}


void VariablePrecisionPosterior::cdf_inplace(std::vector<double>& PH,
                                             bool parallel,
                                             bool adaptive) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->cdf(PH, parallel, adaptive);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->cdf(PH, parallel, adaptive);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->cdf(PH, parallel, adaptive);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->cdf(PH, parallel, adaptive);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->cdf(PH, parallel, adaptive);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::cdf(const std::vector<double>& PH,
                                bool parallel,
                                bool adaptive) const
{
	std::vector<double> result = PH;
	cdf_inplace(result, parallel, adaptive);
	return result;
}


void VariablePrecisionPosterior::tail_inplace(std::vector<double>& PH,
                                              bool parallel,
                                              bool adaptive) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->tail(PH, parallel, adaptive);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->tail(PH, parallel, adaptive);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->tail(PH, parallel, adaptive);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->tail(PH, parallel, adaptive);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->tail(PH, parallel, adaptive);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::tail(const std::vector<double>& PH,
                                 bool parallel,
                                 bool adaptive) const
{
	std::vector<double> result = PH;
	tail_inplace(result, parallel, adaptive);
	return result;
}


void VariablePrecisionPosterior::tail_quantiles_inplace(
           std::vector<double>& quantiles,
           size_t n_chebyshev,
           bool parallel,
           bool adaptive) const
{
	/*
	 * The relevant optionals have been initialized on start.
	 */
	switch (precision){
		case precision_t::WP_DOUBLE:
			posterior_double->tail_quantiles(quantiles, n_chebyshev, parallel,
			                                 adaptive);
			return;
		case precision_t::WP_LONG_DOUBLE:
			posterior_long_double->tail_quantiles(quantiles, n_chebyshev,
			                                      parallel, adaptive);
			return;
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			posterior_float128->tail_quantiles(quantiles, n_chebyshev, parallel,
			                                   adaptive);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			posterior_dec50->tail_quantiles(quantiles, n_chebyshev, parallel,
			                                adaptive);
			return;
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			posterior_dec100->tail_quantiles(quantiles, n_chebyshev, parallel,
			                                 adaptive);
			return;
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}

std::vector<double>
VariablePrecisionPosterior::tail_quantiles(const std::vector<double>& quantiles,
	                                       size_t n_chebyshev,
	                                       bool parallel,
	                                       bool adaptive) const
{
	std::vector<double> result = quantiles;
	tail_quantiles_inplace(result, n_chebyshev, parallel, adaptive);
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
     double p0, double s0, double n0, double v0, double dest_tol
) const
{
	switch (precision){
		case precision_t::WP_DOUBLE:
			return posterior_double->validate(qc_set, p0, s0, n0, v0, dest_tol);
		case precision_t::WP_LONG_DOUBLE:
			return posterior_long_double->validate(qc_set, p0, s0, n0, v0,
			                                       dest_tol);
		case precision_t::WP_FLOAT_128:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD
			return posterior_float128->validate(qc_set, p0, s0, n0, v0,
			                                    dest_tol);
			#endif
		case precision_t::WP_BOOST_DEC_50:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50
			return posterior_dec50->validate(qc_set, p0, s0, n0, v0, dest_tol);
			#endif
		case precision_t::WP_BOOST_DEC_100:
			#ifdef REHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100
			return posterior_dec100->validate(qc_set, p0, s0, n0, v0, dest_tol);
			#endif
		default:
			break;
	}
	throw std::runtime_error("This code is not reached.");
}