/*
 * Point in interval.
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


#ifndef REHEATFUNQ_NUMERICS_INTERVALL_HPP
#define REHEATFUNQ_NUMERICS_INTERVALL_HPP

#include <limits>
#include <cmath>
#include <numerics/limits.hpp>
#include <numerics/functions.hpp>

namespace reheatfunq {
namespace numerics {

namespace rm = reheatfunq::math;

/*
 * Specifiy a point in an interval. Consider the following interval:
 *
 * |---------------|------x--------------|-------------->
 * 0             xmin     x             xmax
 *
 *
 * We now define a class with the following intervals:
 *
 *                 |--ff--|
 *
 * |-------- val ---------|
 *
 *                        |----- fb -----|
 *
 *
 * Where 'ff' is 'from_front' and 'fb' is 'from_back'.
 */
template<typename real>
class PointInInterval {
public:
    constexpr static long double SQRT_EPS
    = std::sqrt(numeric_limits<real>::epsilon());
    real val; // Value
    real from_front; // from front
    real from_back; // from back


    /*
     * Constructors:
     */

    /* Invalid init: */
    PointInInterval() : val(-1), from_front(-1), from_back(-1)
    {}

    PointInInterval(const PointInInterval&) = default;

    constexpr PointInInterval(real val, real from_front, real from_back)
        : val(val), from_front(from_front), from_back(from_back)
    {}


    /*
     * Operators:
     */
    bool operator==(const PointInInterval& other) const {
        bool infr = in_front();
        bool inba = in_back();
        bool oinfr = other.in_front();
        bool oinba = other.in_back();
        if (infr != oinfr || inba != oinba)
            return false;
        if (infr)
            return from_front == other.from_front;
        else if (inba)
            return from_back == other.from_back;
        else
            return val == other.val;
    }

    constexpr operator real() const {
        return val;
    }

    real operator-(const PointInInterval& other) const {
        /* Choose from which coordinates to compute the distance.
            * We do this in a way to retain large precision also at
            * values close to the interval ends: */
        if (in_front() && other.in_front()){
            return from_front - other.from_front;
        }
        if (in_back() && other.in_back()){
            return other.from_back - from_back;
        }
        return val - other.val;
    }

    /*
     * Other functions:
     */

    real distance(const PointInInterval& other) const {
        /* Choose from which coordinates to compute the distance.
            * We do this in a way to retain large precision also at
            * values close to the interval ends: */
        if (in_front() && other.in_front()){
            return rm::abs(from_front - other.from_front);
        }
        if (in_back() && other.in_back()){
            return rm::abs(from_back - other.from_back);
        }
        return rm::abs(val - other.val);
    }

    static PointInInterval<real>
    mean(const PointInInterval<real>& x0,
            const PointInInterval<real>& x1)
    {
        return PointInInterval(
            (x0.val+x1.val) / 2,
            (x0.from_front + x1.from_front) / 2,
            (x0.from_back + x1.from_back) / 2
        );
    }

    static PointInInterval<real>
    mean(const PointInInterval<real>& x0, real w0,
         const PointInInterval<real>& x1, real w1)
    {
        return PointInInterval(
            w0 * x0.val + w1 * x1.val,
            w0 * x0.from_front + w1 * x1.from_front,
            w0 * x0.from_back + w1 * x1.from_back
        );
    }

private:
    bool in_front() const {
        return from_front < SQRT_EPS * from_back;
    }

    bool in_back() const {
        return from_back < SQRT_EPS * from_front;
    }
};

}
}

#endif