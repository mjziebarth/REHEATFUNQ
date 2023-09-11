/*
 * Wrapper around a Python callable object.
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

#ifndef REHEATFUNQ_NUMERICS_PYFUNWRAP_HPP
#define REHEATFUNQ_NUMERICS_PYFUNWRAP_HPP

/*
 * Python include:
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace reheatfunq {
namespace utility {

class PythonFunctionWrapper
{
public:
	PythonFunctionWrapper(PyObject* fun);

	PythonFunctionWrapper(const PythonFunctionWrapper& other);

	~PythonFunctionWrapper();

	double operator()(double x) const;

private:
	PyObject* fun;
};


} // namespace utility
} // namespace reheatfunq

#endif