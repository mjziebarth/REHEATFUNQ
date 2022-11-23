/*
 * Conjugate prior of the gamma distribution due to Miller (1980).
 *
 * Copyright (C) 2021 Deutsches GeoForschungsZentrum GFZ
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
 *
 * Miller (1980):
 */

#ifndef PDTOOLBOX_LL_GAMMA_CONJUGATE_PRIOR_HPP
#define PDTOOLBOX_LL_GAMMA_CONJUGATE_PRIOR_HPP

#include <vector>
#include <array>
#include <memory>
#include <eigenwrap.hpp>
#include <ll_cache.hpp>
#include <gamma_conjugate_prior.hpp>

namespace pdtoolbox {

class GammaConjugatePriorLogLikelihood : public GammaConjugatePriorBase
{
	public:
		typedef std::integral_constant<uint_fast8_t,4> nparams;

		struct ab_t {
			double a;
			double b;
		};
		/*
		 * (1) Creation:
		 */
		GammaConjugatePriorLogLikelihood(double p, double s, double n,
		                                 double v, const double* a,
		                                 const double* b, size_t Nab,
		                                 double nv_surplus_min=1e-3,
		                                 double vmin = 0.1, double amin = 1.0,
		                                 double epsabs=0, double epsrel=1e-10);

		GammaConjugatePriorLogLikelihood(double p, double s, double n,
		                                 double v, const std::vector<ab_t>& ab,
		                                 double nv_surplus_min=1e-3,
		                                 double vmin = 0.1, double amin = 1.0,
		                                 double epsabs=0, double epsrel=1e-10);

		/* Create a unique ptr (for Cython): */
		static std::unique_ptr<GammaConjugatePriorLogLikelihood>
		    make_unique(double p, double s, double n, double v,
			            const double* a, const double* b, size_t Nab,
			            double nv_surplus_min=1e-3, double vmin = 0.1,
			            double amin = 1.0, double epsabs=0,
			            double epsrel = 1e-10);

		/*
		 * (2) The optimization interface:
		 */
		double operator()() const;

		ColumnVector gradient() const;

		constexpr static bool use_hessian = true;
		constexpr static bool boundary_traversal = true;

		SquareMatrix hessian() const;

		ColumnVector parameters() const;

		ColumnVector lower_bound() const;

		ColumnVector upper_bound() const;

		void update(const ColumnVector&);

		void optimize();

		size_t data_count() const;

	private:
		constexpr static double delta = 1e-5;
		const double nv_surplus_min;
		const double vmin;
		double nv;
		double albsum, lbsum, asum, bsum, lgasum;
		double forward[4], backward[4];
		double fw_lp_ls, fw_lp_nv, fw_lp_v, fw_ls_nv, fw_ls_v, fw_nv_v;
		const std::vector<ab_t> ab;
		const double W;

		/* Compute purely data-dependent constants: */
		void init_constants();

		/* Solutions to the numerical integrals, cached: */
		struct integrals_t {
			double lPhi;
			double forward[4];
			double backward[4];
			double fw_lp_ls, fw_lp_nv, fw_lp_v, fw_ls_nv, fw_ls_v, fw_nv_v;
		};

		/* Compute the normalization constant and other integrals: */
		integrals_t integrals() const;

		integrals_t _ints;
		LinearCache<integrals_t, 3> integrals_cache;

};


}

#endif
