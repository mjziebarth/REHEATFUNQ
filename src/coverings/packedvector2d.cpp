/*
 * Testing code of the PackedVector2d class.
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

#include <random>
#include <iostream>
#include <coverings/packedvector2d.hpp>

using reheatfunq::PackedVector2d;

void reheatfunq::test_PackedVector2d(){
	/*
	 * Pseudorandom numbers:
	 */
	std::mt19937_64 rng(9892789734);
	std::uniform_int_distribution<int> ndist(0, 7);
	std::uniform_real_distribution<double> vdist;

	/*
	 * Generate reference data:
	 */
	constexpr size_t N = 1000;
	std::vector<std::pair<double,std::vector<double>>> data(N);
	for (size_t i=0; i<N; ++i){
		/* Generate the random data: */
		data.at(i).first = vdist(rng);
		const size_t m = ndist(rng);
		data.at(i).second.resize(m);
		for (size_t j=0; j<m; ++j){
			data.at(i).second.at(j) = vdist(rng);
		}
	}

	std::cout << "Successfully initialized the data.\n" << std::flush;

	/*
	 * Initialize the packed vector:
	 */
	PackedVector2d<double, double, size_t, true> test(data);

	std::cout << "Successfully initialized the packed vector.\n" << std::flush;

	/*
	 * Compare:
	 */
	for (size_t i=0; i<N; ++i){
		std::cout << "i = (" << i << " / " << N << ")\n" << std::flush;
		if (test[i].first != data.at(i).first)
			throw std::runtime_error("Row data not equal.");
		if (test[i].second.size() != data[i].second.size())
			throw std::runtime_error("Row length not equal.");
		for (size_t j=0; j<data.at(i).second.size(); ++j){
			if (test[i].second[j] != data.at(i).second.at(j))
				throw std::runtime_error("Row elements not equal.");
		}
	}
}
