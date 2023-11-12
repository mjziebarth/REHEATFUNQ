/*
 * Resilience and performance analysis.
 * This file is part of the ziebarth_et_al_2022_heatflow python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
 *               2023 Malte J. Ziebarth
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

#include <include/resilience/resilience.hpp>
#include <include/anomaly/posterior.hpp>
//#include <ziebarth2022a.hpp>
#include <random>
#include <cmath>
#include <type_traits>
#include <array>
#include <omp.h>
#include <stdexcept>

namespace reheatfunq {
namespace resilience {


template<typename dvec_t>
void LS_reference_anomaly_infty_W_m2(const dvec_t& x, dvec_t& q,
                                     double d_km, double Qbar_x2_MW_km)
{
	/*
	 * Compute the reference anomaly from Lachenbruch & Sass (1980)
	 * for a linear increase in source strength with depth.
	 * It is given in eqn. (A23a)
	 *
	 * Parameters:
	 *   x : Distance from fault.
	 *   d : Depth of fault.
	 *
	 * Returns:
	 *   Anomaly heat flow in MW/(km*km) = W/m^2.
	 *
	 * This function is an implementation for infinite times.
	 */
	const double invd = 1.0 / d_km;
	const double Q = 2.0 * Qbar_x2_MW_km * invd;
	const double QoP = Q / M_PI;
	for (size_t i=0; i<x.size(); ++i){
		if (x[i] == 0)
			q[i] = QoP;
		else {
			const double z = x[i] * invd;
			q[i] = QoP * (1.0 - z * std::atan(1.0 / z));
		}
	}
}


template<typename dvec_t>
void LS_anomaly_power_scaled_infty_W_m2(const dvec_t& x, dvec_t& q, double P_MW,
                                        double L_km, double d_km)
{
	/*
	 * Compute the reference anomaly from Lachenbruch & Sass (1980)
	 * scaled to a total power release P on the fault.
	 *
	 * Parameters:
	 *   x  : Perpendicular distance from fault.
	 *   P  : Total power released by the fault.
	 *   L  : Fault length.
	 *   d  : Depth of the fault.
	 *
	 * We distribute the power evenly along the
	 * fault segment length:
	 */
	const double Qbar_x2_MW_km = P_MW / L_km;

	/* Now everything as in Lachenbruch & Sass (1980): */
	LS_reference_anomaly_infty_W_m2(x, q, d_km, Qbar_x2_MW_km);
}


template<typename dvec_t>
void get_lateral_positions_in_circle(dvec_t& x, double radius,
                                     std::mt19937_64& generator)
{
	/*
	 * Draws numbers from a random distribution matching the x coordinates
	 * of points uniformly sampled in a circle
	 */
	for (size_t i=0; i<x.size(); ++i){
		/* Invert the CDF using bisection. */
		const double C = std::uniform_real_distribution()(generator);
		double a,b;
		if (C > 0.5){
			a = 0.0;
			b = 1.0;
		} else {
			a = -1.0;
			b = 0.0;
		}
		double y = 0.5 * (a+b);
		while (std::fabs(b-a) > 1e-13){
			const double f0 = (y * std::sqrt(1.0 - y*y) + std::asin(y)
			                   + 0.5*M_PI) / M_PI - C;
			if (f0 > 0)
				b = y;
			else if (f0 < 0)
				a = y;
			else if (f0 == 0)
				break;
			y = 0.5 * (a+b);
		}
		x[i] = radius * y;
	}
}

/******************************************************************************
 *
 *                             DATA SET GENERATION
 *
 ******************************************************************************/

template<typename dvec_t_>
struct gamma_dist_t {
	double K;
	double T;

	typedef dvec_t_ dvec_t;

	gamma_dist_t(double k, double t) : K(k), T(t)
	{}

	void generate_heat_flow(dvec_t& q_i, std::mt19937_64& rng) const {
		/* Get the gamma random variables: */
		for (size_t i=0; i<q_i.size(); ++i)
			q_i[i] = std::gamma_distribution(K,T)(rng);
	}

};


template<typename dvec_t_>
struct mixture_dist_t {
	double x0;
	double s0;
	double a0;
	double x1;
	double s1;
	double a1;

	typedef dvec_t_ dvec_t;

	mixture_dist_t(double x0, double s0, double a0, double x1, double s1,
	               double a1) : x0(x0), s0(s0), a0(a0), x1(x1), s1(s1), a1(a1)
	{}

	void generate_heat_flow(dvec_t& q_i, std::mt19937_64& gen) const {
		/* Get the mixture random variables: */
		std::uniform_real_distribution<> uni;
		std::normal_distribution<> n0{x0, s0};
		std::normal_distribution<> n1{x1, s1};
		const double w0 = a0 / (a0+a1);
		for (size_t i=0; i<q_i.size(); ++i){
			double q = 0.0;
			for (size_t j=0; j<1000; ++j){
				if (uni(gen) <= w0){
					/* Mixture component 1: */
					q = n0(gen);
					if (q > 0)
						break;
				} else {
					q = n1(gen);
					if (q > 0)
						break;
				}
			}
			if (q <= 0)
				throw std::runtime_error("Could not determine a positive q.");
			q_i[i] = q;
		}
	}
};



template<typename dvec_t, typename dist_t>
void get_synthetic_dataset_mW_m2(dvec_t& x_i, dvec_t& c_i, dvec_t& q_i,
                                 double P_MW,
                                 std::mt19937_64& rng, const dist_t& dist,
                                 double xmax_km,
                                 double L_km, double d_km)
{
	/* Ranges check: */
	if (std::is_same<dvec_t, std::vector<double>>::value){
		if ((x_i.size() != c_i.size()) || (x_i.size() != q_i.size())){
			throw std::runtime_error("Wrong sizes in x_i, c_i, q_i.");
		}
	}
	/*
	 * Computes a synthetic dataset.
	 *
	 * Get the positions:
	 */
	get_lateral_positions_in_circle(x_i, xmax_km, rng);

	/* Get the gamma random variables: */
	dist.generate_heat_flow(q_i, rng);

	/* Get the anomaly: */
	LS_anomaly_power_scaled_infty_W_m2(x_i, c_i, P_MW, L_km, d_km);
	for (size_t i=0; i<x_i.size(); ++i){
		/* We obtained heat flow in W/m^2, but want it in mW/m^2: */
		c_i[i] *= 1e3;

		/* Now add the anomaly profile: */
		q_i[i] += c_i[i];
		c_i[i] /= 1e6 * P_MW;
	}
}



struct fail_t {
    size_t improper;
    size_t proper;
};


template<typename dvec_t>
dvec_t initialize_dvec(size_t N)
{
	return dvec_t();
}

template<>
std::vector<double>
initialize_dvec<std::vector<double>>(size_t N)
{
	return std::vector<double>(N, 0.0);
}


enum veclen_t : unsigned int {
	VARIABLE = 0,
};


template<size_t batch, size_t Nq>
constexpr size_t _tcp_index(size_t i, size_t b, size_t q)
{
	return (i * batch + b) * Nq + q;
}


template<size_t batch, uint_fast8_t Nq, typename dist_t>
fail_t _tcp_iteration_array(size_t N, double P_MW,
                            std::mt19937_64& generator,
                            const dist_t& dist,
                            const typename std::array<double,Nq>& quantiles,
                            typename std::array<double,2*batch*Nq>& res,
                            double p, double s, double n, double v,
                            double amin, double tolerance)
{
	/* Some predefined stuff: */
	constexpr double xmax_km = 80.0;
	constexpr double L_km = 160.0;
	constexpr double d_km = 14.0;

	/* Data arrays: */
	std::vector<double> x_i(N);
	std::vector<double> c_i(N);
	std::vector<double> q_i(N);
	std::vector<double> work_quantiles(Nq);

	/* Data input to Posterior class: */
	std::vector<reheatfunq::anomaly::weighted_sample_t> wsv;
	wsv.emplace_back(1.0);
	wsv[0].sample.reserve(N);

	fail_t failures({0, 0});
	size_t b = 0;
	while (b < batch){
		/* Compute the synthetic data set: */
		double* res_proper = &res.at(_tcp_index<batch,Nq>(0,b,0));
		double* res_improper = &res.at(_tcp_index<batch,Nq>(1,b,0));
		try {
			get_synthetic_dataset_mW_m2(x_i, c_i, q_i, P_MW, generator, dist,
			                            xmax_km, L_km, d_km);
		} catch (...) {
			std::cout << "ERROR in get_synthetic_dataset_mW_m2\n" << std::flush;
			std::fill(res_proper, res_proper+Nq, std::nan(""));
			std::fill(res_improper, res_improper+Nq, std::nan(""));
			++failures.proper;
			++failures.improper;
			continue;
		}

		/* Copy the data from the generated synthetic data set: */
		wsv[0].sample.clear();
		for (size_t i=0; i<N; ++i){
			wsv[0].sample.emplace_back(q_i[i], c_i[i]);
		}

		constexpr size_t BLI_MAX_SPLITS = 100;
		constexpr uint8_t BLI_MAX_REFINEMENTS = 5;

		/* Evaluate the anomalies: */
		try {
			reheatfunq::anomaly::Posterior<long double>
			   post(wsv, p, s, n, v, amin, tolerance,
			        BLI_MAX_SPLITS, BLI_MAX_REFINEMENTS);

			for (uint_fast8_t i=0; i<Nq; ++i){
				work_quantiles[i] = quantiles[i];
			}
			post.tail_quantiles(work_quantiles);
		} catch (std::exception& e) {
			std::cout << "ERROR in Posterior informed\n" << std::flush;
			std::cout << e.what() << "\n" << std::flush;
			std::fill(res_proper, res_proper+Nq, std::nan(""));
			++failures.proper;

			std::cout << std::setprecision(20);
			std::cout << "qc = [\n";
			for (auto qc : wsv[0].sample){
				std::cout << "   [" << qc.q << "," << qc.c << "],\n";
			}
			std::cout << "]\n" << std::flush;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			throw e;
		}

		try {
			reheatfunq::anomaly::Posterior<long double>
			   post(wsv, 1.0, 0.0, 0.0, 0.0, amin, tolerance,
			        BLI_MAX_SPLITS, BLI_MAX_REFINEMENTS);

			for (uint_fast8_t i=0; i<Nq; ++i){
				work_quantiles[i] = quantiles[i];
			}
			post.tail_quantiles(work_quantiles);
		} catch (...) {
			std::cout << "ERROR in Posterior uninformed\n" << std::flush;
			std::fill(res_improper, res_improper+Nq, std::nan(""));
			++failures.improper;
		}

		++b;
	}

	return failures;
}


template<size_t Nq, typename dist_t>
std::optional<std::vector<quantiles_t>>
test_performance_cpp(size_t N, size_t M, double P_MW, const dist_t& dist,
                     const std::array<double,Nq>& quantiles, double PRIOR_P,
                     double PRIOR_S, double PRIOR_N, double PRIOR_V,
                     double amin, bool verbose, bool show_failures, size_t seed,
                     unsigned short nthread, double tolerance)
{
	using namespace std::chrono_literals;

	/*
	 * Tests the performance of the gamma model (with and without prior) for
	 * synthetic data sets that do not stem from a gamma distribution.
	 */
	if (nthread == 0)
		nthread = omp_get_num_procs();

	/* Generate the seed sequence: */
	std::seed_seq seq{seed};
	std::vector<size_t> seeds(nthread);
	seq.generate(seeds.begin(), seeds.end());

	constexpr uint_fast8_t batch = 10;

	size_t improper = 0;
	size_t proper = 0;
	/* Ceiling: */
	const size_t J = M / batch + (M % batch != 0);
	std::vector<quantiles_t> result(M*Nq);
	bool err_thread_num = false;
	#pragma omp parallel num_threads(nthread) shared(result, err_thread_num) \
	        reduction(+ : improper,proper)
	{
		/* Generate the random number generator local to this thread: */
		const int threadid = omp_get_thread_num();
		if (threadid < 0 || static_cast<size_t>(threadid) > seeds.size()){
			std::cerr << "encountered threadid " << threadid << " for a seed "
			             "array of " << seeds.size() << " entries.\n";
			err_thread_num = true;
		}

		if (!err_thread_num){
			const size_t seedi = seeds[threadid];
			std::mt19937_64 generator(seedi);

			#pragma omp for schedule(dynamic)
			for (size_t j=0; j<J; ++j){
				/* Exit by full continuation if any of the threads
				 * encountered an error: */
				if (err_thread_num)
					continue;

				if ((j+1)*batch <= M){
					/*
					 * We've got a full batch left to work on.
					 */
					std::array<double,2*batch*Nq> res;
					fail_t failures =
					    _tcp_iteration_array<batch,Nq>(N, P_MW, generator,
					                                   dist, quantiles, res,
					                                   PRIOR_P, PRIOR_S,
					                                   PRIOR_N, PRIOR_V,
					                                   amin, tolerance);

					/* Set the results: */
					try {
						if (j*batch >= M){
							for (uint_fast8_t k=0; k<M-j*batch; ++k){
								for (uint_fast8_t l=0; l<Nq; ++l){
									size_t m = _tcp_index<batch,Nq>(j,k,l);
									result.at(m).proper
									   = res.at(_tcp_index<batch,Nq>(0,k,l));
									result.at(m).improper
									   = res.at(_tcp_index<batch,Nq>(1,k,l));
								}
							}
						} else {
							for (uint_fast8_t k=0; k<batch; ++k){
								for (uint_fast8_t l=0; l<Nq; ++l){
									size_t m = _tcp_index<batch,Nq>(j,k,l);
									result.at(m).proper
									   = res.at(_tcp_index<batch,Nq>(0,k,l));
									result.at(m).improper
									   = res.at(_tcp_index<batch,Nq>(1,k,l));
								}
							}
						}
					} catch (...) {
						err_thread_num = true;
						continue;
					}
					improper += failures.improper;
					proper += failures.proper;
				} else {
					/*
					 * No full batch left to do.
					 */
					std::array<double,2*Nq> res;
					for (uint_fast8_t i=0; j*batch + i < M; ++i){
						fail_t failures =
						    _tcp_iteration_array<1,Nq>(N, P_MW, generator,
						                               dist, quantiles, res,
						                               PRIOR_P, PRIOR_S,
						                               PRIOR_N, PRIOR_V,
						                               amin, tolerance);

						try {
							for (uint_fast8_t l=0; l<Nq; ++l){
								size_t m = _tcp_index<batch,Nq>(j,0,0) + _tcp_index<1,Nq>(0,i,l);
								result.at(m).proper
								   = res.at(_tcp_index<1,Nq>(0,0,l));
								result.at(m).improper
								   = res.at(_tcp_index<1,Nq>(1,0,l));
							}
						} catch (...) {
							err_thread_num = true;
							continue;
						}
						improper += failures.improper;
						proper += failures.proper;

					}

				}
			}
		}
	}

	if (err_thread_num){
		return std::optional<std::vector<quantiles_t>>();
	}

	return result;
}


std::optional<std::vector<quantiles_t>>
test_performance_1q(size_t N, size_t M, double P_MW,
                    double K, double T, double quantile, double PRIOR_P,
                    double PRIOR_S, double PRIOR_N, double PRIOR_V,
                    double amin, bool verbose, bool show_failures, size_t seed,
                    unsigned short nthread, double tolerance)
{
	std::array<double,1> quant({quantile});
	gamma_dist_t<std::vector<double>> gma(K,T);
	return test_performance_cpp<1>(N, M, P_MW, gma, quant, PRIOR_P,
	                               PRIOR_S, PRIOR_N, PRIOR_V, amin,
	                               verbose, show_failures, seed, nthread,
	                               tolerance);
}


std::optional<std::vector<quantiles_t>>
test_performance_41q(size_t N, size_t M, double P_MW,
                    double K, double T,
                    const std::array<double,41>& quantile, double PRIOR_P,
                    double PRIOR_S, double PRIOR_N, double PRIOR_V,
                    double amin, bool verbose, bool show_failures, size_t seed,
                    unsigned short nthread, double tolerance)
{

	gamma_dist_t<std::vector<double>> gma(K,T);
	return test_performance_cpp<41>(N, M, P_MW, gma, quantile, PRIOR_P,
	                               PRIOR_S, PRIOR_N, PRIOR_V, amin,
	                               verbose, show_failures, seed, nthread,
	                               tolerance);
}

std::optional<std::vector<quantiles_t>>
test_performance_mixture_4q(size_t N, size_t M, double P_MW,
                    double x0, double s0, double a0, double x1, double s1,
                    double a1, const std::array<double,4>& quantiles,
                    double PRIOR_P, double PRIOR_S, double PRIOR_N,
                    double PRIOR_V, double amin, bool verbose,
                    bool show_failures, size_t seed, unsigned short nthread,
                    double tolerance)
{
	mixture_dist_t<std::vector<double>> mix(x0, s0, a0, x1, s1, a1);
	return test_performance_cpp<4>(N, M, P_MW, mix, quantiles, PRIOR_P,
	                               PRIOR_S, PRIOR_N, PRIOR_V, amin, verbose,
	                               show_failures, seed, nthread,
	                               tolerance);

}

std::optional<std::vector<quantiles_t>>
test_performance_mixture_41q(size_t N, size_t M, double P_MW,
                    double x0, double s0, double a0, double x1, double s1,
                    double a1, const std::array<double,41>& quantiles,
                    double PRIOR_P, double PRIOR_S, double PRIOR_N,
                    double PRIOR_V, double amin, bool verbose,
                    bool show_failures, size_t seed, unsigned short nthread,
                    double tolerance)
{
	mixture_dist_t<std::vector<double>> mix(x0, s0, a0, x1, s1, a1);
	return test_performance_cpp<41>(N, M, P_MW, mix, quantiles, PRIOR_P,
	                                PRIOR_S, PRIOR_N, PRIOR_V, amin, verbose,
	                                show_failures, seed, nthread,
	                                tolerance);

}


} // namespace resilience
} // namespace reheatfunq