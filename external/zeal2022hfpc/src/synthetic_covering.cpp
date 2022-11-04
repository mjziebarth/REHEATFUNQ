/*
 * Code to generate a synthetic covering.
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
#include <../include/synthetic_covering.hpp>
#include <../include/mixture_error.hpp>
#include <../include/types.hpp>
#include <algorithm>
#include <omp.h>

//#define TEST_REVIEW_SI_ERROR

using paperheatflow::mixture_t;
using paperheatflow::MixtureErrorGenerator;
using paperheatflow::gamma_params;
using paperheatflow::sample_params_t;
using paperheatflow::covering_t;

template<typename ErrorGenerator>
std::vector<double>
generate_heat_flow_sample(const gamma_params& kt, size_t N, double hf_max,
                          const typename ErrorGenerator::params_t& params,
                          std::shared_ptr<std::mt19937_64>& gen)
{
	std::vector<double> res(N);
	ErrorGenerator mixgen(params, gen);
	for (size_t i=0; i<N; ++i){
		bool success = false;
		while (!success){
			/* Generate heat flow value: */
			double hf = std::gamma_distribution(kt.k, kt.t)(*gen);
			
			/* Add uncertainty: */
			double err = mixgen();
			if (std::uniform_real_distribution()(*gen) <= 0.5){
				hf -= err * hf;
			} else {
				hf += err * hf;
			}

			/* Round: */
			hf = std::round(hf);

			/* Test an error of the first review round.
			 * In the initial Python code used to create
			 * these Monte-Carlo data sets, the values <= 0
			 * were not discarded but instead reassigned to
			 * 1. This lead to a clustering at the singular
			 * value 1, which received all of the probability
			 * mass at negative heat flow values. This had a
			 * large impact if the standard deviation of
			 * the error was large. */
			#ifdef TEST_REVIEW_SI_ERROR
			if (hf <= 0.0)
				hf = 1.0;
			#endif

			/* Cap: */
			if (hf > 0.0 && hf <= hf_max){
				res[i] = hf;
				success = true;
			}
		}
	}

	/* Sort the heat flow values: */
	std::sort(res.begin(), res.end());

	return res;
}


template<typename ErrorGenerator>
covering_t
generate_synthetic_heat_flow_covering(
     const std::vector<sample_params_t>& sample_params, double hf_max,
     const typename ErrorGenerator::params_t& error_params,
     std::shared_ptr<std::mt19937_64>& gen)
{
	covering_t res;
	for (const sample_params_t& sp : sample_params){
		res.push_back(generate_heat_flow_sample<ErrorGenerator>(sp.kt, sp.N,
		                                            hf_max, error_params, gen));
	}
	return res;
}


std::vector<covering_t>
paperheatflow::generate_synthetic_heat_flow_coverings_mixture(
     const std::vector<sample_params_t>& sample_params, size_t N, double hf_max,
     double w0, double x00, double s0, double x10, double s1,
     size_t seed, unsigned short nthread)
{
	/* Generate the seed sequence: */
	std::seed_seq seq{seed};
	std::vector<size_t> seeds(nthread);
	seq.generate(seeds.begin(), seeds.end());

	/* Generate the coverings: */
	mixture_t error_params(w0, x00, s0, x10, s1);
	std::vector<covering_t> res(N);
	#pragma omp parallel num_threads(nthread)
	{
		/* Generate the random number generator local to this thread: */
		const int threadid = omp_get_thread_num();
		if (threadid < 0 || static_cast<size_t>(threadid) > seeds.size())
			throw std::runtime_error("Thread number wrong!");
		const size_t seedi = seeds[threadid];
		std::shared_ptr<std::mt19937_64>
		    gen(std::make_shared<std::mt19937_64>(seedi));

		/* Perform the generation of the converings: */
		#pragma omp for schedule(static)
		for (size_t i=0; i<N; ++i){
			/* Generate the random number generator for this covering: */
			res[i] = generate_synthetic_heat_flow_covering
			            <MixtureErrorGenerator>(sample_params, hf_max,
			                                    error_params, gen);
		}
	}
	
	return res;
}
