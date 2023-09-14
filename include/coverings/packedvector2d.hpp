/*
 * A packed two-dimensional vector with optional row information.
 * Meant to replace constructs such as vector[pair[T0, vector[T1]]].
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

#include <vector>
#include <stdexcept>
#include <algorithm>

#ifndef REHEATFUNQ_COVERINGS_PACKEDVECTOR2D_HPP
#define REHEATFUNQ_COVERINGS_PACKEDVECTOR2D_HPP

namespace reheatfunq {


/*
 *
 *  PointerRange
 *
 */

template<typename T, typename index_t=size_t>
class PointerRange {
public:
	/*
	 * Continuous range iterator interface:
	 */
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T* const_iterator;

	PointerRange(const T* start, const T* end) : _start(start), _end(end)
	{};

	const_iterator begin() const {
		return _start;
	}

	const_iterator end() const {
		return _end;
	}

	const_iterator cbegin() const {
		return _start;
	}

	const_iterator cend() const {
		return _end;
	}

	const T& operator[](size_t i) const {
		if (_start + i >= _end)
			throw std::out_of_range("Attempted to access element out of range "
			                        "in PointerRange.");
		return _start[i];
	}

	index_t size() const {
		return _end-_start;
	}


private:
	const T* _start;
	const T* _end;
};


/*
 *
 *  PackedVector2dIterator
 *
 */

template<typename T, typename rowdata_t, typename index_t,
         bool isvoid=std::is_void_v<rowdata_t>>
struct rettype;

template<typename T, typename rowdata_t, typename index_t>
struct rettype<T, rowdata_t, index_t, true>
{
	typedef PointerRange<T,index_t> value_t;
};

template<typename T, typename rowdata_t, typename index_t>
struct rettype<T, rowdata_t, index_t, false>
{
	typedef std::pair<rowdata_t, PointerRange<T,index_t>>  value_t;
};



template<typename T, typename rowdata_t=void, typename index_t=size_t,
         bool debug=false>
class PackedVector2d;

template<typename T, typename rowdata_t, typename index_t>
class PackedVector2dIterator {
public:
	/*
	 * Iterator interface:
	 */
	typedef index_t difference_type;
	typedef rettype<T, rowdata_t, index_t>::value_t value_type;

	PackedVector2dIterator(const PackedVector2d<T,rowdata_t,index_t>* v,
	                       size_t i)
	   : vec(v), i(i)
	{}

	bool operator==(const PackedVector2dIterator& other) const {
		return (vec == other.vec) && (i == other.i);
	}

	value_type operator*() const {
		return (*vec)[i];
	}

	PackedVector2dIterator& operator++() {
		++i;
		return *this;
	}

	PackedVector2dIterator operator++(int) {
		PackedVector2dIterator res(vec, i);
		++i;
		return res;
	}

private:
	size_t i;
	const PackedVector2d<T, rowdata_t, index_t>* vec;
};


/*
 *
 *  PackedVector2d
 *
 */


template<typename T, typename rowdata_t, typename index_t, bool debug>
class PackedVector2d {
public:
	friend PackedVector2dIterator<T, rowdata_t, index_t>;

	typedef PackedVector2dIterator<T, rowdata_t, index_t> const_iterator;

	PackedVector2d() : storage(0), N(0), index_ptr(nullptr), data_ptr(nullptr)
	{}

	template<typename r = rowdata_t>
	PackedVector2d(
	    std::enable_if_t<!std::is_void_v<r>,
	             const std::vector<std::pair<rowdata_t,std::vector<T>>>&> v)
	   : storage(0), N(v.size()), index_ptr(nullptr), data_ptr(nullptr)
	{
		constexpr size_t istride = sizeof(index_t);

		/* Compute the storage requirement: */
		size_t bytes = (N+1) * istride;
		for (const std::pair<rowdata_t,std::vector<T>>& vi : v){
			bytes += vi.second.size() * TSTRIDE + RDSTRIDE;
		}
		storage.resize(bytes);

		index_ptr = reinterpret_cast<const size_t*>(storage.data());
		data_ptr = storage.data() + (N+1)*istride;

		size_t row = 0;
		size_t flat_id = 0;
		size_t* it = reinterpret_cast<size_t*>(storage.data());
		*it = flat_id;
		for (const std::pair<rowdata_t,std::vector<T>>& vi : v){
			/* Save the row data information: */
			*rowdata_pointer(row) = vi.first;

			/* Fill the data values:: */
			std::copy(vi.second.cbegin(), vi.second.cend(),
			          data_begin_pointer(row));

			/* Mark the end, the next beginning, and advance row: */
			++it;
			flat_id += vi.second.size();
			*it = flat_id;
			++row;
		}
	}


	index_t row_size(index_t i) const {
		if (i >= N)
			throw std::out_of_range("Attempt to access row size out of range "
			                        "in packedvector2d.");
		return index_ptr[i+1] - index_ptr[i];
	}

	template<typename r = rowdata_t>
	std::enable_if_t<!std::is_void_v<r>,
	    std::pair<rowdata_t, PointerRange<T, index_t>>
	>
	operator[](index_t i) const {
		if (i >= N)
			throw std::out_of_range("Attempt to access row out of range in "
			                        "packedvector2d.");
		std::pair<rowdata_t, PointerRange<T, index_t>>
		   ret(*rowdata_pointer(i), pointer_range(i));
		return ret;
	}

	const_iterator begin() const {
		return PackedVector2dIterator<T, rowdata_t, index_t>(this, 0);
	}

	const_iterator end() const {
		return PackedVector2dIterator<T, rowdata_t, index_t>(this, N);
	}


private:
	/* The size allocated before each row storage for the rowdata_t.
	 * sizeof(rowdata_t) if rowdata_t is given (i.e. not void), else 0.
	 */
	constexpr static size_t RDSTRIDE
	   = (std::is_void_v<rowdata_t>) ? 0 : sizeof(rowdata_t);

	constexpr static size_t TSTRIDE = sizeof(T);

	std::vector<char> storage;
	size_t N;
	const size_t* index_ptr;
	char* data_ptr;

	/*************************************************************************
	 * Functions that compute the pointers to the elements.
	 */
	rowdata_t* rowdata_pointer(index_t i){
		char* ptr = data_ptr + i * RDSTRIDE + index_ptr[i] * TSTRIDE;
		if (debug){
			if (ptr >= storage.data() + storage.size())
				throw std::out_of_range("rowdata_pointer");
		}
		return reinterpret_cast<rowdata_t*>(ptr);
	}
	const rowdata_t* rowdata_pointer(index_t i) const {
		const char* ptr = data_ptr + i * RDSTRIDE + index_ptr[i] * TSTRIDE;
		if (debug){
			if (ptr >= storage.data() + storage.size())
				throw std::out_of_range("rowdata_pointer");
		}
		return reinterpret_cast<const rowdata_t*>(ptr);
	}

	inline T* data_begin_pointer(index_t i){
		char* ptr = data_ptr + (i+1) * RDSTRIDE + index_ptr[i] * TSTRIDE;
		if (debug){
			if (ptr >= storage.data() + storage.size())
				throw std::out_of_range("data_begin_pointer");
		}
		return reinterpret_cast<T*>(ptr);
	}
	inline const T* data_begin_pointer(index_t i) const {
		const char* ptr = data_ptr + (i+1) * RDSTRIDE + index_ptr[i] * TSTRIDE;
		if (debug){
			if (ptr > storage.data() + storage.size())
				throw std::out_of_range("data_begin_pointer");
		}
		return reinterpret_cast<const T*>(ptr);
	}

	inline T* data_end_pointer(index_t i){
		const char* ptr = data_ptr + (i+1) * RDSTRIDE + index_ptr[i+1] * TSTRIDE;
		if (debug){
			if (ptr >= storage.data() + storage.size())
				throw std::out_of_range("data_end_pointer");
		}
		return reinterpret_cast<T*>(ptr);
	}
	inline const T* data_end_pointer(index_t i) const {
		const char* ptr = data_ptr + (i+1) * RDSTRIDE + index_ptr[i+1] * TSTRIDE;
		if (debug){
			if (ptr > storage.data() + storage.size())
				throw std::out_of_range("data_end_pointer");
		}
		return reinterpret_cast<const T*>(ptr);
	}

	PointerRange<T, index_t> pointer_range(size_t i) const {
		return PointerRange<T, index_t>(data_begin_pointer(i),
		                                data_end_pointer(i));
	}
};


void test_PackedVector2d();

}

#endif