/*
 * Cache for log-likelihood computations.
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

#ifndef PDTOOLBOX_LL_CACHE_HPP
#define PDTOOLBOX_LL_CACHE_HPP

#include <array>
#include <algorithm>
#include <eigenwrap.hpp>

namespace pdtoolbox {


template<typename T, unsigned int size>
class LinearCache
{
	public:
		typedef typename std::array<T, size>::const_iterator iterator;

		LinearCache();

		iterator find(const ColumnVector& param);

		iterator end() const;

		void put(const ColumnVector& param, const T& val);

	private:
		unsigned int circular_index;
		std::array<ColumnVector, size> keys;
		typename std::array<ColumnVector, size>::const_iterator last_key;
		std::array<T,size> values;
};


template<typename T, unsigned int size>
LinearCache<T,size>::LinearCache() : circular_index(0),
    last_key(keys.cbegin())
{}


template<typename T, unsigned int size>
typename LinearCache<T,size>::iterator
LinearCache<T,size>::find(const ColumnVector& param)
{
	/* Obtain the index (its value is <size>, when nothing is found): */
	size_t index = (std::find(keys.cbegin(), last_key, param) - keys.cbegin());

	/* Establish a 'kind of' least recently used property: */
	if (index != size)
		circular_index = (index + 1) % size;

	return values.begin() + index;
}

template<typename T, unsigned int size>
typename LinearCache<T,size>::iterator
LinearCache<T,size>::end() const
{
	if (last_key == keys.cend())
		return values.cend();

	return values.cbegin() + (last_key - keys.cbegin());
}

template<typename T, unsigned int size>
void LinearCache<T,size>::put(const ColumnVector& param, const T& val)
{
	/* Check whether the key is already in the cache: */
	auto it = std::find(keys.cbegin(), last_key, param);

	if (it != last_key){
		/* Override another element. */
		values[it - keys.cbegin()] = val;
	} else {
		/* Create new cache entry. */
		if (last_key != keys.cend()){
			/* First append to the cache: */
			keys[last_key - keys.cbegin()] = param;
			values[last_key - keys.cbegin()] = val;
			++last_key;
		} else {
			/* Later circulate through the cache: */
			keys[circular_index] = param;
			values[circular_index] = val;
			circular_index = (circular_index + 1) % size;
		}
	}
}


} // end namespace

#endif
