/*
 * Adaptive integrator with partial evaluation based on
 * Simpson's rule.
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


#ifndef REHEATFUNQ_NUMERICS_SIMPSON_HPP
#define REHEATFUNQ_NUMERICS_SIMPSON_HPP

#include <vector>
#include <stack>
#include <cmath>
#include <limits>

#include <thread>
#include <chrono>
#include <iostream>

#include <boost/math/tools/precision.hpp>

#include <numerics/functions.hpp>
#include <numerics/kahan.hpp>
#include <numerics/limits.hpp>

namespace reheatfunq {
namespace numerics {

namespace rm = reheatfunq::math;

/*
 * Simpson's quadrature rule and the underlying
 * quadratic Lagrange interpolation polynomial.
 */
template<typename real>
class SimpsonQuadRule {
public:
    SimpsonQuadRule(real dx, real f1, real f2, real f3)
       : dx(dx), C0(f1), C1((-3*f1 + 4*f2 - f3)/dx),
         C2(2/(dx*dx) * (f1 - 2*f2 + f3)),
         I(dx/6 * (f1 + 4*f2 + f3))
    {
        /* Ensure that the integral using the constants C1, C2, and C3
         * leads to the total integral: */
        real scale = I / integral(dx);
        C0 *= scale;
        C1 *= scale;
        C2 *= scale;
    }

    real integral() const {
        return I;
    }

    /*
     * We can also evaluate the integral of the underlying
     * Lagrange interpolating polynomial to any point within
     * [0,dx]:
     */
    real integral(real x) const {
        return x * (C0 + x * (C1/2 + x * C2/3));
    }

    real density(real x) const {
        return C0 + x * (C1 + x * C2);
    }

    void normalize(real norm) {
        C0 /= norm;
        C1 /= norm;
        C2 /= norm;
        I /= norm;
    }

private:
    real dx;
    real C0;
    real C1;
    real C2;
    real I;
};



/*
 * The integrator: a tree of Simpson rules.
 */
template<typename real, typename point=real, uint8_t min_levels=5>
class SimpsonAdaptiveQuadrature {
public:
    template<typename fun_t>
    SimpsonAdaptiveQuadrature(fun_t fun, point xmin, point xmax,
                              uint8_t max_levels=24)
       : xmin(xmin),
         intervals(adapt(fun, xmin, xmax, max_levels))
    {
    }

    real integral() const {
        /* Full integral. */
        return intervals.back().I0_fw + intervals.back().rule.integral();
    }

    real integral(const point& x, bool backward=false) const {
        /* Early exits: */
        if (x <= xmin){
            if (backward)
                return intervals.front().I0_bw + intervals.front().rule.integral();
            else
                return 0.0;
        }
        if (x >= intervals.back().xmax){
            if (backward)
                return 0.0;
            else
                return intervals.back().I0_fw + intervals.back().rule.integral();
        }

        /* Find the interval. */
        auto it = std::lower_bound(
            intervals.cbegin(),
            intervals.cend(),
            x,
            [](const Leaf& leaf, const point& x) -> bool
            {
                return leaf.xmax < x;
            }
        );

        if (it == intervals.cend())
            throw std::runtime_error("EXCEED AFTER CHECK!");

        point xl = (it == intervals.cbegin()) ? xmin : (it-1)->xmax;

        if (backward)
            return it->I0_bw + it->rule.integral() - it->rule.integral(x - xl);
        else
            return it->I0_fw + it->rule.integral(x - xl);
    }


    real density(const point& x, bool backward=false) const {
        /* Early exits: */
        if (x <= xmin){
            return 0.0;
        }
        if (x >= intervals.back().xmax){
            return 0.0;
        }

        /* Find the interval. */
        auto it = std::lower_bound(
            intervals.cbegin(),
            intervals.cend(),
            x,
            [](const Leaf& leaf, const point& x) -> bool
            {
                return leaf.xmax < x;
            }
        );

        if (it == intervals.cend())
            throw std::runtime_error("EXCEED AFTER CHECK!");

        point xl = (it == intervals.cbegin()) ? xmin : (it-1)->xmax;

        return it->rule.density(x - xl);
    }


private:
    /*
     * The begin of the interval:
     */
    point xmin;

    /*
     * The interval structure.
     */
    struct Leaf {
        point xmax;
        real I0_fw;
        real I0_bw;
        SimpsonQuadRule<real> rule;

        Leaf(point xmax, real I0_fw, real I0_bw, const SimpsonQuadRule<real>& rule)
           : xmax(xmax), I0_fw(I0_fw), I0_bw(I0_bw), rule(rule)
        {}

        Leaf(Leaf&&) = default;
        Leaf(const Leaf&) = default;
    };
    std::vector<Leaf> intervals;


    /*
     * The adaptive routine:
     */
    template<typename fun_t>
    static std::vector<Leaf>
    adapt(fun_t fun, point xmin, point xmax, uint8_t max_levels)
    {
        constexpr long double tol = std::sqrt(numeric_limits<real>::epsilon());
        std::vector<Leaf> done;

        struct todo_t {
            Leaf leaf;
            real f1;
            real f2;
            real f3;
            uint8_t level;

            todo_t(Leaf&& leaf, real f1, real f2, real f3, uint8_t level)
                : leaf(std::move(leaf)), f1(f1), f2(f2), f3(f3), level(level)
            {
            }

            todo_t(todo_t&&) = default;
            todo_t(const todo_t&) = default;
        };
        std::stack<todo_t> todo;

        /* First element: */
        real xl = xmin;
        {
            real f1 = fun(xl);
            real f2 = fun((xmax+xmin)/2);
            real f3 = fun(xmax);
            todo.emplace(
                Leaf(xmax, 0.0, 0.0,
                     SimpsonQuadRule<real>(
                        xmax-xmin, f1,
                        f2, f3)
                    ),
                f1, f2, f3, 1
            );
        }

        while (!todo.empty())
        {
            /* Check the quadrature results obtained when splitting the
             * interval vs the half-interval quadratures of the full
             * interval.
             */
            real dx = todo.top().leaf.xmax - xl;
            real x2 = (xl + todo.top().leaf.xmax) / 2;
            real x12 = (xl + x2) / 2;
            real x23 = (x2 + todo.top().leaf.xmax) / 2;
            real f12 = fun(x12);
            real f23 = fun(x23);
            SimpsonQuadRule<real> sql(dx/2, todo.top().f1, f12, todo.top().f2);
            SimpsonQuadRule<real> sqr(dx/2, todo.top().f2, f23, todo.top().f3);

            real Il = sql.integral();
            real Ir = sqr.integral();
            real I_full = todo.top().leaf.rule.integral();
            real Ic_full = todo.top().leaf.rule.integral(dx/2);

            if ((rm::abs(Il - Ic_full) <= tol * rm::abs(Il)) &&
                (rm::abs(Ir - (I_full - Ic_full)) <= tol * rm::abs(Ir)) &&
                (todo.top().level >= min_levels))
            {
                /* Accept the interval since both sub-integrals are
                 * within tolerance of the value obtained from the root
                 * integrator.
                 */
                done.push_back(todo.top().leaf);
                xl = todo.top().leaf.xmax;
                todo.pop();
            } else if (todo.top().level >= max_levels) {
                /*
                 * Accept the interval because max_levels is exceeded.
                 */
                done.push_back(todo.top().leaf);
                xl = todo.top().leaf.xmax;
                todo.pop();
            } else {
                /* Split the interval. */
                todo_t root(todo.top());
                todo.pop();

                /* First the right interval: */
                todo.emplace(Leaf(root.leaf.xmax, 0.0, 0.0, sqr),
                             root.f2, f23, root.f3,
                             root.level+1);

                /* Second the left interval: */
                todo.emplace(Leaf(x2, 0.0, 0.0, sql),
                             root.f1, f12, root.f2,
                             root.level+1);
            }
        }


        /* For the rare case that the function can be sufficiently
         * described by a second-order polynomial and the code is
         * compiled with min_levels <= 1: */
        if (done.size() == 1)
            return done;


        /* Accumulate the mass, first forward then backwards. */
        auto it = done.begin();
        KahanAdder<real> I;
        I += it->rule.integral();
        while (++it != done.end())
        {
            it->I0_fw = static_cast<real>(I);
            I += it->rule.integral();
        }
        --it;

        /* Normalization: */
        real norm = static_cast<real>(I);

        I = it->rule.integral();
        --it;
        while (it != done.begin())
        {
            it->I0_bw = static_cast<real>(I);
            I += it->rule.integral();
            --it;
        }
        it->I0_bw = static_cast<real>(I);
        I += it->rule.integral();

        /* Ensure unit normalization: */
        it->I0_fw /= norm;
        it->I0_bw /= norm;
        it->rule.normalize(norm);
        while (++it != done.end())
        {
            it->I0_fw /= norm;
            it->I0_bw /= norm;
            it->rule.normalize(norm);
        }

        return done;

    }
};


} // namespace numerics
} // namespace reheatfunq


#endif