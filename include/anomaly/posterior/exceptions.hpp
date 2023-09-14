/*
 * Heat flow anomaly analysis posterior numerics: exceptions.
 *
 * Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2021-2022 Deutsches GeoForschungsZentrum GFZ,
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
 */

#ifndef REHEATFUNQ_ANOMALY_POSTERIOR_EXCEPTIONS_HPP
#define REHEATFUNQ_ANOMALY_POSTERIOR_EXCEPTIONS_HPP

namespace reheatfunq {
namespace anomaly {
namespace posterior {

/* A custom exception type indicating that the integral is out of
 * scale for double precision: */
template<typename real>
class ScaleError : public std::exception
{
public:
	explicit ScaleError(const char* msg, real log_scale)
	    : _lscale(log_scale), msg(msg) {};

	virtual const char* what() const noexcept
	{
		return msg;
	}

	real log_scale() const
	{
		return _lscale;
	};

private:
	real _lscale;
	const char* msg;
};


template<typename real>
class PrecisionError : public std::exception
{
public:
	explicit PrecisionError(const char* msg, real error, real L1)
	    : _error(error), _L1(L1),
	      msg(generate_message(msg, static_cast<double>(error),
	                           static_cast<double>(L1))) {};

	virtual const char* what() const noexcept
	{
		return msg.c_str();
	}

	real error() const
	{
		return _error;
	};

	real L1() const
	{
		return _L1;
	};

	void append_message(const char* ap_msg){
		msg.append("\n");
		msg.append(ap_msg);
	}

private:
	double _error, _L1;
	std::string msg;

	static std::string generate_message(const char* msg, double error,
	                                    double L1)
	{
		std::string smsg("PrecisionError(\"");
		smsg.append(msg);
		smsg.append("\")\n  error: ");
		std::ostringstream os_err;
		os_err << std::setprecision(18);
		os_err << error;
		smsg.append(os_err.str());
		smsg.append("\n     L1: ");
		std::ostringstream os_L1;
		os_L1 << std::setprecision(18);
		os_L1 << L1;
		smsg.append(os_L1.str());
		return smsg;
	}
};

} // namespace posterior
} // namespace anomaly
} // namespace reheatfunq
#endif