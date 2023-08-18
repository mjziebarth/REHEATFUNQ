/*
 * Function evaluation cache.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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

#include <map>
#include <boost/container/flat_map.hpp>
#include <array>
#include <vector>
#include <utility>
#include <cstddef>
#include <algorithm>
#include <functional>
#include <iostream>

#ifndef PDTOOLBOX_FUNCCACHE_HPP
#define PDTOOLBOX_FUNCCACHE_HPP

namespace pdtoolbox {


/*
 * Note: This class is not concurrency-proof and will likely behaving
 *       undesiredly when used from multiple threads.
 */

template<size_t n, typename real>
class SortedCache {
public:
	typedef real real_t;
	typedef std::array<real_t,n> arg_t;
	typedef std::vector<std::pair<const arg_t, real_t>> dump_t;

	SortedCache(std::function<real_t(const arg_t&)> fun)
	   : fun(fun), _cache_size(0)
	{}

	SortedCache(std::function<real_t(const arg_t&)> fun,
	            const dump_t& cache_dump)
	   : fun(fun), cache(cache_dump.cbegin(), cache_dump.cend()),
	     _cache_size(cache_dump.size())
	{}

	real_t operator()(const arg_t& arg) {
		/* Try to find the element: */
		auto it = cache.lower_bound(arg);
		if (it != cache.end() && it->first == arg){
			++_hits;
			return it->second;
		}
		++_misses;

		/* If not, evaluate the function: */
		real_t f = fun(arg);

		/* Emplace the value in the cache using the previously sought
		 * position:
		 */
		cache.emplace_hint(it, arg, f);
		++_cache_size;

		/* Return the function value: */
		return f;
	};

	size_t size() const {
		return cache.size();
	}

	size_t hits() const {
		return _hits;
	}

	size_t misses() const {
		return _misses;
	}

	double hit_rate() const {
		return static_cast<double>(_hits) / static_cast<double>(_hits+_misses);
	}

	/*
	 * Dump and load the cache.
	 */
	dump_t dump() const {
		return dump_t(cache.cbegin(), cache.cend());
	}

	bool operator==(const SortedCache& other) const {
		if (cache.size() != other.cache.size()){
			return false;
		}
		return cache == other.cache;
	}



private:
	/*
	 * The function to call:
	 */
	const std::function<real_t(const arg_t&)> fun;

	/*
	 * The sorted map that acts as a cache:
	 */
	mutable boost::container::flat_map<arg_t,real_t> cache;

	mutable size_t _hits = 0;
	mutable size_t _misses = 0;
	mutable size_t _cache_size;

};



}

#endif