/*
 * Posterior of the volume.
 * NOTE: This development file is not currently in use.
 *
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

#include <../include/volume_posterior.hpp>

#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <stdexcept>


double marginal_posterior_log_unnormed(double P_H, double p, double s, double n,
                                       double v,
                                       const std::vector<double>& qi_mW_m2,
                                       const std::vector<double>& ci_mW_m2);

using paperheatflow::heat_t;
using boost::math::interpolators::cardinal_cubic_b_spline;

void paperheatflow::compute_posterior_volume_parameters(
         const double* efficiency, size_t Neff,
         const double* power_GW, size_t Npow,
         const std::vector<std::vector<double>>& xis,
         const double* weights, size_t Nxi,
         const std::vector<std::vector<heat_t>>& heat_flow_data,
         double prior_p, double prior_s, double prior_n, double prior_v,
         bool renorm_P_H_dimension, bool efficiency_similar,
         double max_efficiency_saf, size_t n_ph, size_t n_interp,
         double* posterior_out, double* mass_out)
{
	/* This method computes the marginalized posterior density
	 * for the parameter space of efficiency and loading rate
	 * with regards to the ENCOS volume.
	 */

	/* Sanity: */
	const size_t Nxi = weights.size();
	for (auto& xi : xis){
		if (xi.size() != Nxi)
			throw std::runtime_error("Size of all xi vectors has to coincide "
			                         "with size of the weight vector.");
	}
	const size_t Nreg = heat_flow_data.size();
	if (xis.size() != Nreg)
		throw std::runtime_error("There has to be one data vector for each "
		                         "vector in `xis`.");


	/* Create an interpolator for the posteriors: */
	std::vector<double> P_H_max;
	std::vector<cardinal_cubic_b_spline<double>> interpolators;
	for (const std::vector<heat_t>& hfd : heat_flow_data){
		/* Make sure we have data: */
		if (hfd.size() == 0)
			continue;

		/* For interfacing with the posterior, transfer qi and ci
		 * to two individual vectors:
		 */
		std::vector<double> qi_mW_m2;
		std::vector<double> ci_mW_m2;

		/* Maximum power.
		 * First find the first heat flow data point which is
		 * affected by the anomaly: */
		double PHmax;
		auto it = hfd.begin();
		while (it != hfd.end() && it->c_1_m2 == 0){
			qi_mW_m2.push_back(it->q_mW_m2);
			ci_mW_m2.push_back(it->c_1_m2);
			++it;
		}
		if (it == hfd.end())
			continue;
		PHmax = it->q_mW_m2 / it->c_1_m2;
		for (auto it = hfd.begin()+1; it != hfd.end(); ++it){
			if (it->c_1_m2 == 0)
				continue;
			PHmax = std::min(PHmax, it->q_mW_m2 / it->c_1_m2);
			qi_mW_m2.push_back(it->q_mW_m2);
			ci_mW_m2.push_back(it->c_1_m2);
		}
		P_H_max.push_back(PHmax);

		/* Evaluate the posterior: */
		std::vector<double> eval(n_interp);
		double dPH = PHmax / (n_interp - 1);
		for (size_t i=0; i<n_interp; ++i){
			eval[i] = marginal_posterior_log_unnormed(i * dPH, prior_p, prior_s,
			                                          prior_n, prior_v,
			                                          hfd.q_mW_m2
			                                          qi_mW_m2, ci_mW_m2);
		}

		/* Create the interpolator: */
		interpolators.emplace_back(eval.begin(), eval.end(), 0.0, dPH);
	}

	/* Now iterate over the branches of xi values and compute the P_H
	 * integration:
	 */
	if (Nreg > 1){
//		#pragma omp parallel for
//		for (size_t i=0; i<Npow; ++i){
//			for (size_t j=0; j<Neff; ++j){
//				/* We have one more dimension, the xi dimension: */
//				std::vector<double> log_p_ij(Nxi, 0.0);
//				double max_log_p_ij = -std::numeric_limits<double>::infinity();
//				for (size_t k=0; k<Nxi; ++k){
//					/* Iterate over all regions that contribute.
//					 * The information from the regions is independent,
//					 * hence we multiply the probability across regions: */
//					for (size_t l=0; l<Nreg; ++k){
//						const double start = xis[l][k].xi * (1.0 / max_efficiency_saf)
//						for (size_t m=0; m<n_ph; ++m){
//
//						}
//					}
//				}
//			}
//		}
		throw std::runtime_error("Number of regions > 1: not implemented.");
	} else if (Nreg == 1) {
		#pragma omp parallel for
		for (size_t i=0; i<Npow; ++i){
			for (size_t j=0; j<Neff; ++j){
				const double P = 1e9 * power_GW[i];
				const double eta = efficiency[j];
				const double end_min = (1.0 - eta) * P;
				/* We have one more dimension, the xi dimension: */
				std::vector<double> log_p_ij(Nxi, 0.0);
				double max_log_p_ij = -std::numeric_limits<double>::infinity();
				for (size_t k=0; k<Nxi; ++k){
					/* Iterate over all regions that contribute.
					 * The information from the regions is independent,
					 * hence we multiply the probability across regions: */
					const double start
					    = xis[0][k] * (1.0 / max_efficiency_saf - 1.0)
					                * eta * P;
					if (start >= end){
						/* Not valid. */

					} else {
						/* Integrate. */
						std::vector<double> posterior(n_ph);
						for (size_t l=0; l<n_ph; ++l){
							posterior[l] =
						}
					}

				}
			}
		}
	}
    posterior = np.zeros((xis.size,*eta.shape))
    end = (1-eta)*P
    for i,xiw in enumerate(zip(xis, weights)):
        if i % 100 == 0:
            print("i=",i)
        xi, w = xiw
        # Compute the range of energy release on the San Andreas fault
        # at the integration points:
        start = xi * (1/max_efficiency_saf - 1) * eta * P
        mask_zero = start >= end
        end = np.maximum(start, end)
        P_H = np.linspace(start, end, n_ph,  axis=-1)

        # Evaluate the posterior:
        post_loc = post(P_H)

        pl_max = post_loc.max(axis=-1)
        post_loc -= pl_max[:,:,np.newaxis]
        np.exp(post_loc, out=post_loc)
        post_loc = post_loc.sum(axis=-1)
        post_loc[np.isinf(pl_max)] = 0.0
        post_loc[mask_zero] = 0.0
        np.log(post_loc, out=post_loc)
        post_loc += pl_max
        if not renorm_P_H_dimension:
            post_loc += np.log(end-start)
        posterior[i,...] = post_loc
        posterior[i,...] += np.log(w)
}

/*
def compute_posterior_volume_parameters(efficiency, power_GW, xis, weights,
                                        prior_p, prior_s, prior_n, prior_v, q_i, c_i,
                                        renorm_P_H_dimension=True,
                                        efficiency_similar=False,
                                        max_efficiency_saf=1.0,
                                        n_ph=2000, n_interp=10000, verbose_plot=False):
    """
    This method computes the marginalized posterior density
    for the parameter space of efficiency and loading rate
    with regards to the ENCOS volume.
    """
    # Create the parameter grid and the mass matrix:
    eta, P = np.meshgrid(efficiency, 1e9*power_GW, indexing='ij')
    d_eta = np.diff(efficiency)
    d_P   = np.diff(power_GW)
    mass_eta = np.zeros_like(efficiency)
    mass_eta[:-1] = 0.5*d_eta
    mass_eta[1:] += 0.5*d_eta
    mass_P = np.zeros_like(power_GW)
    mass_P[:-1] = 0.5*d_P
    mass_P[1:] += 0.5*d_P
    mass = mass_eta[:,np.newaxis] * mass_P[np.newaxis,:]

    # Enforce ordering:
    if not np.all(d_eta > 0) or not np.all(d_P > 0):
        raise RuntimeError("Please give efficiency and power in ascending order.")


    # Create an interpolator for the posterior:
    P_H_max_W = np.min(q_i/c_i)
    P_H_interp_W = np.linspace(0, P_H_max_W, n_interp)
    mplu = marginal_posterior_log_unnormed(P_H_interp_W, prior_p, prior_s, prior_n, prior_v, q_i, c_i)
    post_interp = interp1d(P_H_interp_W, mplu)
    def post(P_H):
        post_ = P_H
        mask = P_H <= P_H_max_W
        post_[mask] = post_interp(P_H[mask])
        post_[~mask] = -np.inf
        return post_

    if verbose_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(P_H_interp_W, mplu)



    # Now iterate over the branches of xi values and compute the P_H
    # integration:
    posterior = np.zeros((xis.size,*eta.shape))
    end = (1-eta)*P
    for i,xiw in enumerate(zip(xis, weights)):
        if i % 100 == 0:
            print("i=",i)
        xi, w = xiw
        # Compute the range of energy release on the San Andreas fault
        # at the integration points:
        start = xi * (1/max_efficiency_saf - 1) * eta * P
        mask_zero = start >= end
        end = np.maximum(start, end)
        P_H = np.linspace(start, end, n_ph,  axis=-1)

        # Evaluate the posterior:
        post_loc = post(P_H)

        pl_max = post_loc.max(axis=-1)
        post_loc -= pl_max[:,:,np.newaxis]
        np.exp(post_loc, out=post_loc)
        post_loc = post_loc.sum(axis=-1)
        post_loc[np.isinf(pl_max)] = 0.0
        post_loc[mask_zero] = 0.0
        np.log(post_loc, out=post_loc)
        post_loc += pl_max
        if not renorm_P_H_dimension:
            post_loc += np.log(end-start)
        posterior[i,...] = post_loc
        posterior[i,...] += np.log(w)

    # Return the posterior from log space and contract the xi dimension:
    post_max = posterior.max()
    posterior -= post_max
    np.exp(posterior, out=posterior)
    posterior = posterior.sum(axis=0)

    # Norm the posterior:
    posterior /= (mass*posterior).sum()

    return posterior, mass
*/
