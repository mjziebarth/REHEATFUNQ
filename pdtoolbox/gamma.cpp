/*
 * Gamma distribution maximum likelihood estimate.
 */

#include <gamma.hpp>
#include <cmath>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

using boost::math::digamma, boost::math::trigamma;

namespace pdtoolbox {

gamma_mle_t compute_gamma_mle(const double mean_x, const double mean_logx,
                              const double amin)
{
	/*
	 * Initial guess:
	 * From: Minka, Thomas P. (2002): "Estimating a Gamma distribution."
	 */
	const double s = std::log(mean_x) - mean_logx;
	double a = (3.0 - s + std::sqrt((s - 3.0)*(s - 3.0) + 24.0*s)) / (12.0*s);

	/*
	 * Newton-Raphson iteration:
	 */
	double a1;
	a1 = a - (std::log(a) - digamma(a) - s) \
	                / (1.0/a - trigamma(a));
	size_t i = 0;
	while (std::fabs(a1 - a) > 1e-13 && i < 20){
		a = a1;
		a1 = a - (std::log(a) - digamma(a) - s) \
		          / (1.0/a - trigamma(a));
		a1 = std::max(a1, amin);
		++i;
	}

	/*
	 * Set and return the final results:
	 */
	gamma_mle_t res;
	res.a = a1;
	res.b = res.a / mean_x;

	return res;
}



}
