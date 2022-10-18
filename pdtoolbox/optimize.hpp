/*
 * Gradient descend for likelihood optimization.
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
 */

#ifndef PDTOOLBOX_OPTIMIZE_HPP
#define PDTOOLBOX_OPTIMIZE_HPP

#include <matrix.hpp>

#include <cmath>
/*
 * Levenberg-Marquardt with geodetic acceleration following
 * Transtrum & Sethna (2012)
 */

#include <iostream>



/* DEBUG
 * If this define is set, newton_optimize will output debug information
 * about the trajectory it takes to "test.out" in the running directory. */
//#define PDTOOLBOX_DEBUG

#ifdef PDTOOLBOX_DEBUG
#include <fstream>
#endif


namespace pdtoolbox {



/*
 *
 * Higher dimensional Newton method:
 *
 * Uses the Armijo condition from
 *   L. Armijo (1966): "Minimization of Functions Having Lipschitz Continuous
 *       First Partial Derivatives", Pac. J. Math. 16(1)
 *
 */
template<typename LogLikelihood, typename D,
         std::enable_if<!std::is_same<D,DN>::value,int>::type=0>
bool newton_optimize(LogLikelihood& ll, double g = 0.5,
                     const double gstep_down = 2.0,
                     const double gstep_up = 1.3, const double gmax = 1.0,
                     const size_t nmax=200, double armijo=0.5,
                     double armijo_gradient = 0.1)
{
	/* Parameter vector: */
	double cost = -ll();
	ColumnVector<D> p(ll.parameters());
	const ColumnVector<D> lb(ll.lower_bound());
	const ColumnVector<D> ub(ll.upper_bound());

	/* Step size and max linesearch: */
	const size_t max_linesearch = 20;

	/* DEBUG: */
	#ifdef PDTOOLBOX_DEBUG
	std::ofstream out;
	out.open("test.out");
	out.precision(20);
	bool line_end_written = false;
	#endif


	bool early_exit = false;
	for (size_t i=0; i<nmax; ++i){
		/* Depending on whether the Likelihood supports Hessian computation
		 * or not, we will compute the direction differently. */
		ColumnVector<D> direction;
		bool use_hessian = LogLikelihood::use_hessian;

		/* Compute gradient and Hessian: */
		const ColumnVector<D> grad(-ll.gradient());

		if (LogLikelihood::use_hessian)
		{
			/* Use Hessian and gradient: */
			const SquareMatrix<D> H(-ll.hessian());

			/* Debug output: */
			#ifdef PDTOOLBOX_DEBUG
			line_end_written = false;
			for (int i=0; i<D::value; ++i)
				out << p[i] << ",";
			for (int i=0; i<D::value; ++i)
				out << grad[i] << ",";
			for (int i=0; i<D::value; ++i)
				for (int j=i; j<D::value; ++j)
					out << H(i,j) << ",";
			out << g << "," << cost << ","
				<< H.positive_definite();
			#endif

			/* Depending on whether the function is convex, we choose
			 * the step direction from the Newton method or according
			 * to the steepest descent: */
			use_hessian = H.positive_semidefinite();
			auto propose_step
			   = [&](const SquareMatrix<D>& H, const ColumnVector<D>& grad)
			     -> ColumnVector<D>
			{
				/* If H positive definite, i.e. the function is convex,
				 * perform Newton step: */
				if (use_hessian)
					return -SquareMatrix<D>::solve(H, grad);

				/* Otherwise, propose gradient direction: */
				ColumnVector<D> dir(-grad);

				/* If we want to do the boundary traversal, we need to project
				 * the direction: */
				if (LogLikelihood::boundary_traversal){
					/* First we need to project the gradient if we are at the
					 * boundary: */
					for (typename D::value_type j=0; j<D::value; ++j){
						if (p[j] == lb[j] && dir[j] < 0)
							dir[j] = 0.0;
						else if (p[j] == ub[j] && dir[j] > 0)
							dir[j] = 0.0;
					}
				}

				/* Not using the Hessian is not entirely true - we do
				 * use its information to scale the parameters.
				 * Scale by the Hessian: */
				double dnrm = 1.0 / std::sqrt(H.diagonal().norm2());
				for (typename D::value_type j=0; j<D::value; ++j){
					dir[j] /= std::fmax(1e-17, std::fabs(H(j,j)) * dnrm);
				}

				return dir / std::sqrt(dir.norm2());
			};
			direction = propose_step(H, grad);

		} else {
			/* Use only gradient. */
			if (LogLikelihood::boundary_traversal){
				/* First we need to project the gradient if we are at the
				 * boundary: */
				direction = -grad;
				for (typename D::value_type j=0; j<D::value; ++j){
					if (p[j] == lb[j] && direction[j] < 0)
						direction[j] = 0.0;
					else if (p[j] == ub[j] && direction[j] > 0)
						direction[j] = 0.0;
				}
				const double nrm = std::sqrt(grad.norm2());
				if (nrm == 0.0)
					/* Early exit. */
					early_exit = true;
				direction /= nrm;
			} else
				/* Simple normalized gradient: */
				direction = -(grad / std::sqrt(grad.norm2()));

			/* Debug output: */
			#ifdef PDTOOLBOX_DEBUG
			line_end_written = false;
			for (int i=0; i<D::value; ++i)
				out << p[i] << ",";
			for (int i=0; i<D::value; ++i)
				out << grad[i] << ",";
			out << "0,0,0,0," << g << "," << cost << ","
			    << "0";
			#endif
		}

		/* If we use the Hessian, limit g: */
		if (use_hessian)
			g = std::fmin(g, gmax);

		/* Line search: */
		for (size_t j=0; j<max_linesearch; ++j){
			/* Check early exit: */
			if (LogLikelihood::boundary_traversal && early_exit)
				break;

			/* Evaluate the new parameter step: */
			const ColumnVector<D> dp(g*direction);
			ColumnVector<D> p1(p + dp);

			/* Bounds check: */
			bool in_bounds = true;
			if (LogLikelihood::boundary_traversal){
				/* First identify whether any parameter is newly on the
				 * boundary: */
				typename D::value_type jmin = -1;
				double gmin = g;
				for (typename D::value_type j=0; j<D::value; ++j){
					if (dp[j] == 0)
						continue;
					if (p1[j] <= lb[j]){
						// There might be the case that multiple boundaries
						// are reached within the same step. In these cases, we
						// want to keep advancing on the gradient line, i.e.
						// Set the boundary value only for the first parameter
						// to arrive at the boundary - this means shortening
						// the step length for this step.
						const double gj = std::abs((lb[j]-p[j])/direction[j]);
						if (gj <= gmin || in_bounds)
						{
							jmin = j;
							gmin = gj;
						}
						// We are not anymore within bounds:
						in_bounds = false;

					} else if (p1[j] >= ub[j]) {
						// Same as above:
						const double gj = std::abs((ub[j]-p[j])/direction[j]);
						if (gj < gmin || in_bounds)
						{
							jmin = j;
							gmin = gj;
						}
						// We are not anymore within bounds:
						in_bounds = false;
					}
				}
				/* Now if we are out of bounds, shorten the step: */
				if (!in_bounds){
					for (typename D::value_type j=0; j<D::value; ++j){
						if (j == jmin){
							p1[j] = (p1[j] <= lb[j]) ? lb[j] : ub[j];
						} else {
							p1[j] = std::max(std::min(p[j] + gmin*direction[j],
							                          ub[j]), lb[j]);
						}
					}
				}
			} else {
				for (typename D::value_type j=0; j<D::value; ++j){
					if (p1[j] < lb[j] || p1[j] > ub[j]){
						g /= gstep_down;
						in_bounds = false;
						break;
					}
				}
				if (!in_bounds)
					continue;
			}

			ll.update(p1);
			const double cost_1 = -ll();

			/* Evaluate the Armijo condition.
			 * Empirically, we use the Armijo constraint only when the change
			 * in parameters is large compared to the numerical fluctuations
			 * (i.e. |dp| / |p| > 3e-16).
			 * If this condition is not fulfilled, Armijo constraint might
			 * run the algorithm into fluctuation purgatory. But then, it is
			 * likely that we are already quite close to the solution,
			 * hence we use only a simple improvement check. */
			const double armijo_delta = (armijo==0.0) ? 0.0 :
				                        (use_hessian) ? armijo*(dp*grad)
			                                 : armijo_gradient * (dp*grad);
			if (std::isnan(cost_1) || cost_1 >= cost + armijo_delta) {
				/* Retract update: */
				ll.update(p);
				/* Decrease stepsize: */
				g /= gstep_down;

				/* If the stepsize is already small enough that parameters are
				 * equal to double precision, exit:
				 * (In the first exit, we relax the Armijo constraint)*/
				if ((p-p1).norm2() < 9e-32 * p.norm2()){
					if (armijo == 0.0){
						early_exit = true;
					}
					armijo = 0.0;
					#ifdef PDTOOLBOX_DEBUG
					line_end_written = true;
					out << "," << j << ",0," << cost_1 << "," << dp[0]
					    << "," << dp[1] <<"\n";
					#endif
					break;
				}

				continue;
			}

			/* If we are here, step is acepted! */
			p = p1;
			cost = cost_1;

			/* Increase stepsize: */
			g *= gstep_up;

			/* Exit linesearch: */
			#ifdef PDTOOLBOX_DEBUG
			line_end_written = true;
			out << "," << j << ",0," << cost_1 << "," << dp[0]
				<< "," << dp[1] <<"\n";
			#endif
			break;
		}

		/* Make sure the line is finished: */
		#ifdef PDTOOLBOX_DEBUG
		if (!line_end_written){
			out << "," << max_linesearch << ",0," << cost << ",0,0\n";
		}
		#endif

		if (early_exit)
			break;
	}

	/* DEBUG: */
	#ifdef PDTOOLBOX_DEBUG
	out.close();
	#endif

	/* Return success status (true if succeeded): */
	return early_exit;

}


} // End namespace.

#endif
