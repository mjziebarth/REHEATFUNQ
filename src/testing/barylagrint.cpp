/*
 * Test code for the Barycentric Lagrange interpolator.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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

#include <numerics/barylagrint.hpp>
#include <iomanip>
#include <cmath>

namespace reheatfunq {
namespace testing {

using reheatfunq::numerics::PiecewiseBarycentricLagrangeInterpolator;
using reheatfunq::numerics::PointInInterval;

void test_barycentric_lagrange_interpolator()
{
	auto fun = [](const PointInInterval<double>& x) -> double
	{
		return std::exp(-x.val);
	};

	const double xmax = 5.0;
	PiecewiseBarycentricLagrangeInterpolator<double> bli(fun, 0.0, xmax, 1e-8,
	                                                     0.0, 0.0, 1.0);

	constexpr size_t N = 897389;
	for (size_t i=0; i<N; ++i){
		PointInInterval<double> xi(i * xmax / (N-1),
		                         i * xmax / (N-1),
		                         (N - 1 - i) * xmax / (N-1));
		double y_ref = fun(xi);
		double y = bli(xi);
		if (std::abs(y - y_ref) > 1e-8 * y_ref)
			throw std::runtime_error("Failed test_barycentric_"
			                         "lagrange_interpolator: exp(-x)");
	}


	/*
	 * Function 2:
	 * f(x) = sin(x)**2
	 */
	auto fun2 = [](long double x) -> long double
	{
		double sx = std::sin(x);
		return sx * sx;
	};
	PiecewiseBarycentricLagrangeInterpolator<double> bli2(fun2, 0.0, xmax,
	                                                      1e-10, 0.0, 0.0, 1.0,
	                                                      200, 8);
	for (size_t i=0; i<N; ++i){
		PointInInterval<double> xi(i * xmax / (N-1),
		                         i * xmax / (N-1),
		                         (N - 1 - i) * xmax / (N-1));
		double y_ref = fun2(xi);
		double y = bli2(xi);
		double tol = std::max(1e-7, 1e-16/(y_ref/xi.val));
		if ((std::abs(y - y_ref) > tol * y_ref) && (y_ref > 0)){
			std::cerr << std::setprecision(20);
			std::cerr << "i     = " << i << " / " << N << "\n";
			std::cerr << "xi    = " << xi << "\n";
			std::cerr << "y     = " << y << "\n";
			std::cerr << "y_ref = " << y_ref << "\n";
			std::cerr << "dy    = " << y - y_ref << " (" << (y-y_ref) / y_ref
			          << ")\n";
			throw std::runtime_error("Failed test_barycentric_"
			                         "lagrange_interpolator: sin^2(x).");
		}
	}


	/*
	 * Function 3:
	 * f(x) = Theta(x - 2.0)
	 */
	auto fun3 = [](double x) -> double
	{
		if (x > 2.0)
			return 1.0;
		return 0.0;
	};
	PiecewiseBarycentricLagrangeInterpolator<long double>
	    bli_ld(fun3, 0.0, xmax, 1e-11, 1e-11, 0.0, 1.0, 100, 9);
	for (size_t i=0; i<N; ++i){
		PointInInterval<long double> xi(i * xmax / (N-1),
		                                i * xmax / (N-1),
		                                (N - 1 - i) * xmax / (N-1));
		double y_ref = fun3(xi);
		double y = bli_ld(xi);
		if (std::abs(y - y_ref) > 1e-6){
			std::cerr << "Failed test_barycentric_lagrange_interpolator: Theta(x-2).\n";
			std::cerr << "xi    = " << xi << "\n";
			std::cerr << "y     = " << y << "\n";
			std::cerr << "y_ref = " << y_ref << "\n";
			std::cerr << "dy    = " << y - y_ref << " (" << (y-y_ref) / y_ref
			          << ")\n";
			std::cerr << "This is expected due to the harsh discontinuity. If discontinuity "
			             "detection finds its way back to the interpolator, this problem may "
			             "be resolved.\n";
			break;
		}
	}


	/*
	 * Function 4:
	 * f(x) = exp(-(x-2.0)**2 / (2 * 0.5**2))
	 */
	auto fun4 = [](double x) -> double
	{
		double dx = x - 2.0;
		return std::exp(-0.5 * dx*dx / (0.5*0.5));
	};
	bli = PiecewiseBarycentricLagrangeInterpolator<double>(fun4, 0.0, xmax,
	                                                       1e-8);
	for (size_t i=0; i<N; ++i){
		PointInInterval<double> xi(i * xmax / (N-1),
		                         i * xmax / (N-1),
		                         (N - 1 - i) * xmax / (N-1));
		double y_ref = fun4(xi);
		double y = bli(xi);
		if (std::abs(y - y_ref) > 1e-7 * y_ref){
			throw std::runtime_error("Failed test_barycentric_"
			                         "lagrange_interpolator: "
			                         "exp(-(x-2)**2 / (2 * 0.5**2)).");
		}
	}
}

}
}
