/*
 * Simple explicit matrix code.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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

#ifndef PDTOOLBOX_MATRIX_HPP
#define PDTOOLBOX_MATRIX_HPP

#include <array>
#include <vector>
#include <vector.hpp>
#include <stdexcept>
#include <algorithm>

// We use LAPACK to test some linear algebraic properties
// of the matrices:
#include <lapacke.h>

namespace pdtoolbox {

template<typename D>
class RowSpecifiedMatrix;

template<typename D>
class ColumnSpecifiedMatrix;

template<typename D>
class SquareMatrix;

template<typename D>
class ColumnVector;



template<typename D>
class RowVector : public Vector<D>
{
	public:
		RowVector(RowVector<D>&&) = default;
		RowVector operator*(const SquareMatrix<D>&) const;

		ColumnVector<D> T() const;
};

template<typename D>
class ColumnVector : public Vector<D>
{
	public:
		/* Empty initialization only for D != DN; */
		template<typename T=D>
		ColumnVector(std::enable_if<!std::is_same<T,DN>::value,
		                            size_t>::type=D::value)
		{};

		/* Initialization with explicit size for D == DN; */
		template<typename T=D>
		ColumnVector(std::enable_if<std::is_same<T,DN>::value,
		                            size_t>::type);



		ColumnVector(Vector<D>&&);
		ColumnVector(const Vector<D>&);

		SquareMatrix<D> operator*(const RowVector<D>&) const;

		double operator*(const ColumnVector<D>&) const;

		ColumnVector<D> operator*(double d) const;

		ColumnVector<D> operator/(double d) const;

		RowVector<D> transpose() const;
		RowVector<D> T() const;
};

template<typename D>
ColumnVector<D>::ColumnVector(Vector<D>&& v) : Vector<D>::Vector(std::move(v))
{}

template<typename D>
ColumnVector<D>::ColumnVector(const Vector<D>& v)
   : Vector<D>::Vector(v)
{}

template<typename D>
double ColumnVector<D>::operator*(const ColumnVector& v) const
{
	return Vector<D>::operator*(v);
}

template<typename D>
ColumnVector<D> ColumnVector<D>::operator*(double d) const
{
	return Vector<D>::operator*(d);
}

template<typename D>
ColumnVector<D> ColumnVector<D>::operator/(double d) const
{
	return Vector<D>::operator*(1.0 / d);
}


template<typename D>
ColumnVector<D> operator*(double d, const ColumnVector<D>& v)
{
	return v * d;
}




/* Square matrix.
 * Depending on whether dimension is fixed of variable,
 * we use vector or array to compile:
 */
template<typename D,size_t storage=0>
class SquareMatrixBase
{
protected:
	std::array<double,storage> data;
};


template<>
class SquareMatrixBase<DN>
{
protected:
	std::vector<double> data;
};


template<typename D>
class SquareMatrix : public SquareMatrixBase<D,D::value*D::value>{
	public:
		SquareMatrix();
		//SquareMatrix(SquareMatrix<D> &&);

		static SquareMatrix<D> diagonal(const Vector<D>&);
		static SquareMatrix<D> diagonal(double d);

		Vector<D> diagonal() const;

		static SquareMatrix<D> outer_product(const Vector<D>&, const Vector<D>&);

		double operator()(size_t i, size_t j) const;
		double& operator()(size_t i, size_t j);

		/* Inverse and solving linear equations: */
		SquareMatrix inv() const&;
		SquareMatrix&& inv() &&;

		static ColumnVector<D> solve(const SquareMatrix<D>& A,
		                             const ColumnVector<D>& b);

		// Check whether a matrix is positive definite:
		bool positive_definite() const;
		bool positive_semidefinite() const;
		bool negative_semidefinite() const;
		bool symmetric() const;

		/* Arithmetic operators: */
		SquareMatrix<D> operator*(const SquareMatrix<D>&) const;

		SquareMatrix<D> operator*(double) const &;
		SquareMatrix<D> && operator*(double) &&;

		SquareMatrix<D> operator+(const SquareMatrix<D>&) const &;
		SquareMatrix<D>&& operator+(const SquareMatrix<D>&) &&;
		SquareMatrix<D>&& operator+(SquareMatrix<D>&&) const &;
		SquareMatrix<D>&& operator+(SquareMatrix<D>&&) &&;

		SquareMatrix<D> operator-() const &;
		SquareMatrix<D>&& operator-() &&;

		ColumnVector<D> operator*(const ColumnVector<D>&) const;


		Vector<D> operator*(const Vector<D>& v) const;

	private:
		size_t size() const;
};

template<typename D>
size_t SquareMatrix<D>::size() const
{
	static_assert(!std::is_same<D,DN>::value);
	return D::value;
}


template<typename D>
SquareMatrix<D>::SquareMatrix()
{
	SquareMatrixBase<D,D::value*D::value>::data.fill(0.0);
}

template<typename D>
SquareMatrix<D> SquareMatrix<D>::diagonal(const Vector<D>& d)
{
	static_assert(!std::is_same<D,DN>::value);
	SquareMatrix<D> m;
	for (size_t i=0; i<D::value; ++i){
		m.data[(D::value+1)*i] = d[i];
	}
	return m;
}

template<typename D>
SquareMatrix<D> SquareMatrix<D>::diagonal(double d)
{
	static_assert(!std::is_same<D,DN>::value);
	SquareMatrix<D> m;
	for (size_t i=0; i<D::value; ++i){
		m.data[(D::value+1)*i] = d;
	}
	return m;
}

template<typename D>
Vector<D> SquareMatrix<D>::diagonal() const
{
	static_assert(!std::is_same<D,DN>::value);
	Vector<D> v;
	for (size_t i=0; i<D::value; ++i){
		v[i] = SquareMatrixBase<D,D::value*D::value>::data[(D::value+1)*i];
	}
	return v;
}
template<typename D>
SquareMatrix<D> SquareMatrix<D>::outer_product(const Vector<D>& v0,
                                               const Vector<D>& v1)
{
	static_assert(!std::is_same<D,DN>::value);
	SquareMatrix<D> m;
	for (size_t i=0; i<D::value; ++i){
		for (size_t j=0; j<D::value; ++j){
			m.data[D::value*i + j] = v0[i] * v1[j];
		}
	}
	return m;
}

template<typename D>
double SquareMatrix<D>::operator()(size_t i, size_t j) const
{
	return SquareMatrixBase<D,D::value*D::value>::data[size()*i + j];
}

template<typename D>
double& SquareMatrix<D>::operator()(size_t i, size_t j)
{
	return SquareMatrixBase<D,D::value*D::value>::data[size()*i + j];
}






/* ***************************************************
 *
 *               Arithmetic operators.
 *
 * ****************************************************/

/*
 *   MULTIPLICATION
 */
template<typename D>
SquareMatrix<D> SquareMatrix<D>::operator*(const SquareMatrix<D>& m) const
{
	SquareMatrix<D> sm;
	//for (typename D::value_type i=0; i<D::value; ++i){
	// 	for (typename D::value_type j=0; j<D::value; ++j){
	// 		double d = 0;
	// 		for (typename D::value_type k=0; k<D::value; ++k)
	// 			d += operator()(i,k) * csm(k,j);
	// 		sm(i,j) = d;
	// 	}
	// }
	for (typename D::value_type i=0; i<D::value; ++i){
		for (typename D::value_type k=0; k<D::value; ++k){
			const double tik = operator()(i,k);
			for (typename D::value_type j=0; j<D::value; ++j){
				sm(i,j) += tik * csm(k,j);
			}
		}
	}
	return sm;
}

template<typename D>
SquareMatrix<D>&& SquareMatrix<D>::operator*(double d) &&
{
	for (double& x : this->data){
		x *= d;
	}
	return std::forward<SquareMatrix<D>>(*this);
}

template<typename D>
SquareMatrix<D> SquareMatrix<D>::operator*(double d) const &
{
	SquareMatrix<D> sm;
	for (typename D::value_type i=0; i<D::value*D::value; ++i){
		sm.data[i] = d * this->data[i];
	}
	return sm;
}

/* Create operators for left-multiplying doubles: */
template<typename D>
SquareMatrix<D> operator*(double d, const SquareMatrix<D>& m)
{ return m * d;
}
template<typename D>
SquareMatrix<D>&& operator*(double d, SquareMatrix<D>&& m)
{ return std::forward<SquareMatrix<D>>(m).operator*(d);
}



/*
 *   ADDITION
 */
template<typename D>
SquareMatrix<D>&& SquareMatrix<D>::operator+(const SquareMatrix<D>& m) &&
{
	for (typename D::value_type i=0; i<D::value*D::value; ++i){
		this->data[i] += m.data[i];
	}
	return std::forward<SquareMatrix<D>>(*this);
}

template<typename D>
SquareMatrix<D>&& SquareMatrix<D>::operator+(SquareMatrix<D>&& m) &&
{
   for (typename D::value_type i=0; i<D::value*D::value; ++i){
	   this->data[i] += m.data[i];
   }
   return std::forward<SquareMatrix<D>>(*this);
}

template<typename D>
SquareMatrix<D>&& SquareMatrix<D>::operator+(SquareMatrix<D>&& m1) const &
{
	for (typename D::value_type i=0; i<D::value*D::value; ++i){
		m1.data[i] += this->data[i];
	}
	return std::forward<SquareMatrix<D>>(m1);
}

template<typename D>
SquareMatrix<D> SquareMatrix<D>::operator+(const SquareMatrix<D>& m) const &
{
	SquareMatrix<D> m2;
	for (typename D::value_type i=0; i<D::value*D::value; ++i){
		m2.data[i] = this->data[i] + m.data[i];
	}
	return m2;
}


/*
 *   UNARY NEGATION:
 */
template<typename D>
SquareMatrix<D>&&  SquareMatrix<D>::operator-() &&
{
	for (double& d : this->data){
		d = -d;
	}
	return std::forward<SquareMatrix<D>>(*this);
}

template<typename D>
ColumnVector<D> SquareMatrix<D>::operator*(const ColumnVector<D>& cv) const
{
	ColumnVector<D> res;
	for (size_t i=0; i<D::value; ++i){
		double d = 0;
		for (size_t j=0; j<D::value; ++j){
			d += this->operator()(i,j) * cv[j];
		}
		res[i] = d;
	}
	return res;
}



/*************************************************
 *
 *    Testing properties of the matrices.
 *
 *************************************************/
template<typename D>
bool SquareMatrix<D>::positive_definite() const
{
	static_assert(!std::is_same<D,DN>::value);
	// Use LAPACK to test whether Cholesky-factorization works.
	constexpr size_t N = D::value;
	typedef SquareMatrixBase<D,D::value*D::value> Base;
	/* Temporary copy: */
	double Abuf[N*N];
	std::copy(Base::data.begin(), Base::data.end(), Abuf);
	lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'u', N, Abuf, N);
	return info == 0;
}

template<typename D>
bool SquareMatrix<D>::symmetric() const
{
	static_assert(!std::is_same<D,DN>::value);
	for (typename D::value_type i=0; i<D::value-1;++i){
		for (typename D::value_type j=i+1; j<D::value; ++j){
			if (this->operator()(i,j) != this->operator()(j,i))
			{
				return false;
			}
		}
	}
	return true;
}








/*
 *
 *      HalfSpecifiedMatrix
 *
 */

template<typename D>
class HalfSpecifiedMatrix {
	protected:
		HalfSpecifiedMatrix(size_t N);
		double operator()(size_t unspec, size_t spec) const;
		double& operator()(size_t unspec, size_t spec);
		size_t unspec_size() const;

	private:
		size_t N;
		std::vector<double> vec;
};

template<typename D>
HalfSpecifiedMatrix<D>::HalfSpecifiedMatrix(size_t N) : N(N), vec(N*D::value)
{
	static_assert(!std::is_same<D,DN>::value);
}

template<typename D>
double& HalfSpecifiedMatrix<D>::operator()(size_t unspec, size_t spec)
{
	return vec[unspec*D::value + spec];
}

template<typename D>
double HalfSpecifiedMatrix<D>::operator()(size_t unspec, size_t spec) const
{
	return vec[unspec*D::value + spec];
}

template<typename D>
size_t HalfSpecifiedMatrix<D>::unspec_size() const
{
	return N;
}


/*
 *
 *       RowSpecifiedMatrix
 *
 */

template<typename D>
class RowSpecifiedMatrix : public HalfSpecifiedMatrix<D> {
	friend class ColumnSpecifiedMatrix<D>;
	public:
		RowSpecifiedMatrix(RowSpecifiedMatrix<D>&&);
		RowSpecifiedMatrix(size_t N);

		double operator()(size_t i, size_t j) const;
		double& operator()(size_t i, size_t j);

		ColumnSpecifiedMatrix<D> T() const;

		SquareMatrix<D> operator*(const ColumnSpecifiedMatrix<D>&) const;

		ColumnVector<D> operator*(const ColumnVector<DN>&) const;

	private:
		RowSpecifiedMatrix(const HalfSpecifiedMatrix<D>&);
		RowSpecifiedMatrix(HalfSpecifiedMatrix<D>&&);

};

template<typename D>
RowSpecifiedMatrix<D>::RowSpecifiedMatrix(size_t N)
   : HalfSpecifiedMatrix<D>(N)
{}

template<typename D>
RowSpecifiedMatrix<D>::RowSpecifiedMatrix(HalfSpecifiedMatrix<D>&& hsm)
    : HalfSpecifiedMatrix<D>(std::move(hsm))
{}

template<typename D>
RowSpecifiedMatrix<D>::RowSpecifiedMatrix(const HalfSpecifiedMatrix<D>& hsm)
    : HalfSpecifiedMatrix<D>(hsm)
{}

template<typename D>
SquareMatrix<D>
RowSpecifiedMatrix<D>::operator*(const ColumnSpecifiedMatrix<D>& csm) const
{
	const size_t N = this->unspec_size();
	if (N != csm.unspec_size())
		throw std::runtime_error("Trying to multiply RowSpecifiedMatrix with "
		                         "ColumnSpecifiedMatrix of incompatible size.");
	SquareMatrix<D> sm;
	for (size_t i=0; i<D::value; ++i){
		for (size_t j=0; j<D::value; ++j){
			double d = 0;
			for (size_t k=0; k<N; ++k)
				d += operator()(i,k) * csm(k,j);
			sm(i,j) = d;
		}
	}
	return sm;
}

template<typename D>
ColumnVector<D>
RowSpecifiedMatrix<D>::operator*(const ColumnVector<DN>& cv) const
{
	const size_t N = this->unspec_size();
	if (N != cv.size())
		throw std::runtime_error("Trying to multiply RowSpecifiedMatrix with "
		                         "ColumnVector of incompatible size.");
	ColumnVector<D> res;
	for (size_t i=0; i<D::value; ++i){
		double d = 0;
		for (size_t j=0; j<N; ++j){
			d+= operator()(i,j) * cv[j];
		}
		res[i] = d;
	}
	return res;
}



template<typename D>
double RowSpecifiedMatrix<D>::operator()(size_t i, size_t j) const
{
	return HalfSpecifiedMatrix<D>::operator()(j,i);
}


template<typename D>
double& RowSpecifiedMatrix<D>::operator()(size_t i, size_t j)
{
	return HalfSpecifiedMatrix<D>::operator()(j,i);
}


/*
 *
 *       ColumnSpecifiedMatrix
 *
 */

template<typename D>
class ColumnSpecifiedMatrix : public HalfSpecifiedMatrix<D> {
	friend class RowSpecifiedMatrix<D>;
	public:
		ColumnSpecifiedMatrix(ColumnSpecifiedMatrix<D>&&);
		ColumnSpecifiedMatrix(size_t N);

		double operator()(size_t i, size_t j) const;
		double& operator()(size_t i, size_t j);

		RowSpecifiedMatrix<D> T() const;

		ColumnVector<D> operator*(const ColumnVector<DN>&) const;
};

template<typename D>
ColumnSpecifiedMatrix<D>::ColumnSpecifiedMatrix(size_t N)
   : HalfSpecifiedMatrix<D>(N)
{}

template<typename D>
ColumnSpecifiedMatrix<D>::ColumnSpecifiedMatrix(ColumnSpecifiedMatrix<D>&& m)
   : HalfSpecifiedMatrix<D>(m)
{}


template<typename D>
double ColumnSpecifiedMatrix<D>::operator()(size_t i, size_t j) const
{
	return HalfSpecifiedMatrix<D>::operator()(i,j);
}


template<typename D>
double& ColumnSpecifiedMatrix<D>::operator()(size_t i, size_t j)
{
	return HalfSpecifiedMatrix<D>::operator()(i,j);
}

template<typename D>
RowSpecifiedMatrix<D> ColumnSpecifiedMatrix<D>::T() const
{
	return RowSpecifiedMatrix<D>(*this);
}









/***************************************************************/
/*                      SPECIALIZATION                         */
/***************************************************************/

template<typename D>
SquareMatrix<D> Vector<D>::outer(const Vector& other) const
{
	return SquareMatrix<D>::outer_product(*this, other);
}




}

#endif
