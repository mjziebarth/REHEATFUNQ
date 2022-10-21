/*
 * Gamma distribution maximum likelihood estimate.
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