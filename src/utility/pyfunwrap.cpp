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

#include <utility/pyfunwrap.hpp>
#include <stdexcept>

using reheatfunq::utility::PythonFunctionWrapper;

PythonFunctionWrapper::PythonFunctionWrapper(PyObject* func)
   : fun(nullptr)
{
	/* If no object given, raise Error: */
	if (!func)
		throw std::runtime_error("Creating PythonFunctionWrapper with NULL "
		                         "function reference");

	/* Acquire GIL: */
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();

	/* First obtain a strong reference: */
	Py_INCREF(func);

	/* Release GIL */
	PyGILState_Release(gstate);

	/* Now set the value: */
	fun = func;
}

PythonFunctionWrapper::PythonFunctionWrapper(const PythonFunctionWrapper& pfw)
   : fun(nullptr)
{
	/* Acquire GIL: */
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();

	/* First obtain a strong reference: */
	Py_INCREF(pfw.fun);

	/* Release GIL */
	PyGILState_Release(gstate);

	/* Now set the value: */
	fun = pfw.fun;
}

PythonFunctionWrapper::~PythonFunctionWrapper()
{
	/* Ensure destruction of strong reference: */
	if (fun){
		/* Acquire GIL: */
		PyGILState_STATE gstate;
		gstate = PyGILState_Ensure();

		Py_DECREF(fun);

		/* Release GIL */
		PyGILState_Release(gstate);
	}
}

double PythonFunctionWrapper::operator()(double x) const
{
	/* Acquire GIL: */
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();

	/* Check the error state: */
	const PyObject* err = PyErr_Occurred();
	if (err){
		/* Release GIL */
		PyGILState_Release(gstate);
		throw std::runtime_error("An error has occurred before entering "
		                         "the call of PythonFunctionWrapper.");
	}

	/* Call the function: */
	PyObject* x_py = PyFloat_FromDouble(x);
	if (!x_py){
		/* Release GIL and throw error: */
		PyGILState_Release(gstate);
		throw std::runtime_error("An error occurred trying to construct the "
		                         "floating point argument in "
		                         "PythonFunctionWrapper call.");
	}
	PyObject* res = PyObject_CallOneArg(fun, x_py);
	/* Clear the arguments first: */
	Py_DECREF(x_py);
	x_py = nullptr;

	/* Check whether any error occurred: */
	if (!res){
		/* Release GIL */
		PyGILState_Release(gstate);
		throw std::runtime_error("An error occurred trying to call the "
		                         "function given to PythonFunctionWrapper.");
	}

	/* Convert to double: */
	const double result = PyFloat_AsDouble(res);

	/* Check whether error occurred: */
	err = PyErr_Occurred();

	/* Release GIL */
	PyGILState_Release(gstate);

	/* Raise on error: */
	if (err){
		throw std::runtime_error("Converting the result of the function call "
		                         "in PythonFunctionWrapper to float failed.");
	}

	return result;
}


