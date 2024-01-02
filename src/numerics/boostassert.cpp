/*
 * Boost assertion capture.
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
 * [1] Ziebarth, Anderson, von Specht, Heidbach, Cotton (in prep.)
 */

#include <stdexcept>
#include <string>


/* Transforming boost asserts into runtime errors: */
namespace boost {

void assertion_failed(char const * expr, char const * function,
                      char const * file, long line)
{
	std::string s0(expr);
	std::string s1(function);
	std::string s2(file);
	std::string s3 = std::to_string(line);
	std::string msg("Error: '");
	msg += s0;
	msg += std::string("' in function '");
	msg += s1;
	msg += std::string("' in file '");
	msg += s2;
	msg += std::string("' in line ");
	msg += s3;
	msg += std::string(".");
	throw std::runtime_error(msg);
}

void assertion_failed_msg(char const * expr, char const* msgc,
                          char const * function, char const * file, long line)
{
	std::string s0(expr);
	std::string s1(msgc);
	std::string s2(function);
	std::string s3(file);
	std::string s4 = std::to_string(line);
	std::string msg("Error: '");
	msg += s0;
	msg += std::string("'\n\twith message '");
	msg += s1;
	msg += std::string("'\n\tin function '");
	msg += s2;
	msg += std::string("'\n\tin file '");
	msg += s3;
	msg += std::string("'\n\tin line ");
	msg += s4;
	msg += std::string(".");
	throw std::runtime_error(msg);
}

}
