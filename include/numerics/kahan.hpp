/*
 * Kahan summation.
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

#ifndef REHEATFUNQ_NUMERICS_KAHAN_HPP
#define REHEATFUNQ_NUMERICS_KAHAN_HPP

#include <type_traits>

namespace reheatfunq {
namespace numerics {


#pragma GCC push_options
#pragma GCC optimize("-fno-associative-math")
template<typename real>
real kahan_sum(const double* x, size_t N)
{
	real S = 0.0;
	real corr = 0.0;
	for (size_t i=0; i<N; ++i){
		real dS = static_cast<real>(x[i]) - corr;
		real next = S + dS;
		corr = (next - S) - dS;
		S = next;
	}
	return S;
}

template<typename real>
real kahan_sum(const std::vector<real>& x)
{
	real S = 0.0;
	real corr = 0.0;
	for (const real& xi : x){
		real dS = xi - corr;
		real next = S + dS;
		corr = (next - S) - dS;
		S = next;
	}
	return S;
}

template<typename function, typename container,
         typename arg = container::value_type,
         typename real=std::invoke_result_t<function,arg>>
real kahan_sum(const container& x, function fun)
{
	real S = 0.0;
	real corr = 0.0;
	for (const arg& ai : x){
		const real xi = fun(ai);
		real dS = xi - corr;
		real next = S + dS;
		corr = (next - S) - dS;
		S = next;
	}
	return S;
}

template<typename function, typename real=std::invoke_result_t<function,size_t>>
real kahan_sum(size_t N, function fun)
{
	real S = 0.0;
	real corr = 0.0;
	for (size_t i=0; i<N; ++i){
		real dS = static_cast<real>(fun(i)) - corr;
		real next = S + dS;
		corr = (next - S) - dS;
		S = next;
	}
	return S;
}

template<typename real>
class KahanAdder {
public:
	KahanAdder(real S0 = 0.0) : S(S0), corr(0.0)
	{}

	KahanAdder& operator+=(real x) {
		real dS = x - corr;
		real next = S + dS;
		corr = (next - S) - dS;
		S = next;
		return *this;
	}

	KahanAdder& operator+=(const KahanAdder& other) {
		*this += -other.corr;
		*this += other.S;
		return *this;
	}

	operator real() const {
		return S;
	}


private:
	real S;
	real corr;

};


#pragma GCC pop_options


#pragma omp declare reduction(+ : KahanAdder<double> : \
                              omp_out += omp_in) \
            initializer (omp_priv=omp_orig)

#pragma omp declare reduction(+ : KahanAdder<long double> : \
                              omp_out += omp_in) \
            initializer (omp_priv=omp_orig)

}
}


#endif