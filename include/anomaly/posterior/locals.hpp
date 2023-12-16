/*
 * Code for computing the gamma conjugate posterior modified for heat
 * flow anomaly as described by Ziebarth & von Specht [1].
 * This is a basic data structure that lists variables used in the REHEATFUNQ
 * anomaly posterior code.
 * This code is an alternative implementation of the code found in
 * `ziebarth2022a.cpp`.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum GFZ,
 *               2022-2023 Malte J. Ziebarth
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
 * [1] Ziebarth, M. J. and von Specht, S.: REHEATFUNQ 1.4.0: A model for
 *     regional aggregate heat flow distributions and anomaly quantification,
 *     EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-222, 2023.
 */


#ifndef REHEATFUNQ_ANOMALY_POSTERIOR_LOCALS_HPP
#define REHEATFUNQ_ANOMALY_POSTERIOR_LOCALS_HPP

/*
 * REHEATFUNQ includes:
 */
#include <anomaly/posterior/args.hpp>
#include <numerics/functions.hpp>
#include <numerics/kahan.hpp>

/*
 * General includes:
 */
#include <limits>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <utility>
#include <cstddef>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>

namespace reheatfunq {
namespace anomaly {
namespace posterior {

struct qc_t {
	const double q;
	const double c;

	qc_t(double q, double c) : q(q), c(c) {}
};

namespace rm = reheatfunq::math;
namespace rn = reheatfunq::numerics;


template<typename real>
class Locals
{
public:
	/* Prior parameters (potentially updated): */
	real lp;
	real ls;
	real n;
	real v;
	real amin;
	real Qmax;
	std::vector<real> ki;
	std::array<real,4> h;
	real w;
	real lh0;
	real l1p_w;
	real lv;

	/*
	 * The main interface to this class.
	 * It performs three computations before it passes to the private
	 * constructor. This way, we can keep the class immutable.
	 */
	Locals(const std::vector<qc_t>& qc, arg<const real>::type p,
	       arg<const real>::type s, arg<const real>::type n,
	       arg<const real>::type v, arg<const real>::type amin,
	       double dest_tol)
	   : Locals(
	        /* Pass through some of the parameters:  */
	        qc, p, s, n, v, amin, dest_tol,
	        /* A = s + sum(qi) */
	        s + rn::kahan_sum(qc, get_q),
	        /* B = sum(ci) */
	        rn::kahan_sum(qc, [](const qc_t& qc)->real {
	                              return qc.c;
	                          }),
	        /* Compute Qmax and its index imax */
	        compute_Qmax(qc)
	      )
	{}

	/*
	 * Initialize from previously calculated parameters:
	 */
	Locals(arg<const real>::type lp, arg<const real>::type ls,
	       arg<const real>::type n, arg<const real>::type v,
	       arg<const real>::type amin, arg<const real>::type Qmax,
	       std::vector<real>&& ki, const std::array<real,4>& h,
	       arg<const real>::type w, arg<const real>::type lh0,
	       arg<const real>::type l1p_w, arg<const real>::type lv)
	   : lp(lp), ls(ls), n(n), v(v), amin(amin), Qmax(Qmax),
	     ki(std::move(ki)), h(h), w(w), lh0(lh0), l1p_w(l1p_w),
	     lv(lv)
	{}

	template<typename istream>
	Locals(istream& in) : ki(0) {
		in.get(&lp, sizeof(real));
		in.get(&ls, sizeof(real));
		in.get(&n, sizeof(real));
		in.get(&v, sizeof(real));
		in.get(&amin, sizeof(real));
		in.get(&Qmax, sizeof(real));
		size_t N_ki;
		in.get(&N_ki, sizeof(size_t));
		ki.resize(N_ki);
		for (size_t i=0; i<N_ki; ++i){
			in.get(&ki[i], sizeof(real));
		}
		for (uint_fast8_t i=0; i<4; ++i){
			in.get(&h[i], sizeof(real));
		}
		in.get(&w, sizeof(real));
		in.get(&lh0, sizeof(real));
		in.get(&l1p_w, sizeof(real));
		in.get(&lv, sizeof(real));
	}

	Locals() {};


	template<typename ostream>
	void write(ostream& out) const {
		// Locals:
		out.write(&lp, sizeof(real));
		out.write(&ls, sizeof(real));
		out.write(&n, sizeof(real));
		out.write(&v, sizeof(real));
		out.write(&amin, sizeof(real));
		out.write(&Qmax, sizeof(real));
		size_t N_ki = ki.size();
		out.write(&N_ki, sizeof(size_t));
		for (real k : ki){
			out.write(&k, sizeof(real));
		}
		for (real hi : h){
			out.write(&hi, sizeof(real));
		}
		out.write(&w, sizeof(real));
		out.write(&lh0, sizeof(real));
		out.write(&l1p_w, sizeof(real));
		out.write(&lv, sizeof(real));
	}


private:
	static std::pair<real,size_t> compute_Qmax(const std::vector<qc_t>&);
	static std::vector<real> compute_ki(const std::vector<qc_t>& qc);
	static std::array<real,4> compute_h(const std::vector<real>& ki,
	                                    size_t imax);
	static real compute_lqsum(const std::vector<qc_t>& qc);
	static std::vector<real> compute_ki(const std::vector<qc_t>& qc,
	                                    arg<const real>::type Qmax);

	Locals(const std::vector<qc_t>& qc, arg<const real>::type p,
	       arg<const real>::type s, arg<const real>::type n,
	       arg<const real>::type v_, arg<const real>::type amin,
	       double dest_tol, real A, real B, std::pair<real,size_t> Qimax)
	 : lp(rm::log(p) + compute_lqsum(qc)),
	   ls(rm::log(A)),
	   n(n+qc.size()),
	   v(v_+qc.size()),
	   amin(amin),
	   Qmax(Qimax.first),
	   ki(compute_ki(qc, Qmax)),
	   h(compute_h(ki, Qimax.second)),
	   w(B * Qmax / A),
	   lh0(rm::log(h[0])),
	   l1p_w(rm::log1p(-w)),
	   lv(rm::log(v))
	{}

	static real get_q(const qc_t& qc) {
		return qc.q;
	}
};

/*
 * Implementations:
 */

template<typename real>
std::pair<real,size_t>
Locals<real>::compute_Qmax(const std::vector<qc_t>& qc)
{
	size_t imax = 0;
	real Qmax = std::numeric_limits<real>::infinity();
	for (size_t i=0; i<qc.size(); ++i){
		if (qc[i].q <= 0)
			throw std::runtime_error("At least one qi is zero or negative and "
			                         "has hence left the model definition "
			                         "space.");
		real Q = static_cast<real>(qc[i].q) / static_cast<real>(qc[i].c);
		if (Q > 0 && Q < Qmax){
			Qmax = Q;
			imax = i;
		}
	}
	if (rm::isinf(Qmax))
		throw std::runtime_error("Found Qmax == inf. The model is not "
		                         "well-defined. This might happen if all "
		                         "ci <= 0, i.e. heat flow anomaly has a "
		                         "negative or no impact on all data points.");

	/* Make sure that we don't have the same largest element twice: */
	for (size_t i=0; i<qc.size(); ++i){
		real Q = static_cast<real>(qc[i].q) / static_cast<real>(qc[i].c);
		if (Q == Qmax && i != imax)
			throw std::runtime_error("Found two data points for which q_i/c_i "
			                         "is equal to Qmax. This is not accounted "
			                         "for in the current large-z "
			                         "approximation.");
	}

	return std::make_pair(Qmax, imax);
}


template<typename real>
std::vector<real>
Locals<real>::compute_ki(const std::vector<qc_t>& qc,
                         arg<const real>::type Qmax)
{
	std::vector<real> ki(qc.size());
	for (size_t i=0; i<qc.size(); ++i)
		ki[i] = (qc[i].c * Qmax) / qc[i].q;

	for (size_t i=0; i<qc.size(); ++i){
		if (rm::isinf(ki[i]) || rm::isnan(ki[i])){
			std::string msg = "Coefficient k[";
			msg += std::to_string((int)i);
			msg += "] is ";
			msg += std::to_string(static_cast<long double>(ki[i]));
			msg += ". We had q=";
			msg += std::to_string(qc[i].q);
			msg += " and c=";
			msg += std::to_string(qc[i].c);
			msg += ".";
			throw std::runtime_error(msg);
		}
	}
	return ki;
}

template<typename real>
real
Locals<real>::compute_lqsum(const std::vector<qc_t>& qc)
{
	auto get_lq =
	[](const qc_t& qci) -> real
	{
		return rm::log(static_cast<const real>(qci.q));
	};
	return rn::kahan_sum(qc, get_lq);
}

template<typename real>
std::array<real,4>
Locals<real>::compute_h(const std::vector<real>& ki, size_t imax)
{
	/* Compute the coefficients for the large-z (small-y) Taylor expansion: */
	std::array<real,4> h;
	h[0] = 1.0;
	h[1] = 0.0;
	h[2] = 0.0;
	h[3] = 0.0;
	for (size_t i=0; i<ki.size(); ++i){
		if (i == imax)
			continue;
		const real d0 = 1.0 - ki[i];
		h[3] = d0 * h[3] + ki[i] * h[2];
		h[2] = d0 * h[2] + ki[i] * h[1];
		h[1] = d0 * h[1] + ki[i] * h[0];
		h[0] *= d0;
	}
	/*
	 * Check the coefficients for errors:
	 */
	for (uint_fast8_t i=0; i<4; ++i){
		if (rm::isinf(h[i]) || rm::isnan(h[i])){
			std::string msg = "Coefficient h[";
			msg += std::to_string((int)i);
			msg += "] is ";
			msg += std::to_string(static_cast<long double>(h[i]));
			msg += ".";
			throw std::runtime_error(msg);
		}
	}
	return h;
}


} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq

#endif