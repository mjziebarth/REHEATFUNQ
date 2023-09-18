# Test some C++ numeric backend code exposed through the `reheatfunq._testing`
# module (if compiled).
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2023 Malte J. Ziebarth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from math import sin, log, inf

from reheatfunq._testing.barylagrint \
     import _test_barycentric_lagrange_interpolator, \
            barycentric_lagrange_interpolate, \
            BarycentricLagrangeInterpolator

def test_barycentric_lagrange_0():
    def fun(x):
        return sin(x) ** 2

    from datetime import datetime
    t0 = datetime.now()
    bli = BarycentricLagrangeInterpolator(fun, 0.0, 5.0)
    t1 = datetime.now()
    x = np.linspace(0, 5, 98193)
    t2 = datetime.now()
    y = bli(x)
    t3 = datetime.now()
    y_ref = np.array([fun(xi) for xi in x])
    t4 = datetime.now()

    print("setting up the interpolator took", (t1-t0).total_seconds())
    print("interpolating took", (t3-t2).total_seconds())
    print("evaluating took", (t4-t3).total_seconds())

#
# The following is some test code the can be used to inspect some of the
# internal workings of the barycentric Lagrange interpolator.
#
#    def find_minima(samples):
#        n = samples.shape[0]
#        minima = []
#        for i in range(1,n-1):
#            fi = samples[i,1]
#            if samples[i-1,1] > fi and samples[i+1,1] > fi:
#                minima.append(i)
#        return minima
#
#    samples = bli.samples()
#
#    import matplotlib.pyplot as plt
#    fig = plt.figure(figsize=(10,5))
#    ax = fig.add_subplot(121)
#    ax.plot(x, y)
#    ax.plot(x, y_ref, linewidth=0.7)
#
#    ax = fig.add_subplot(122)
#    ax.set_xlim(3.1, 3.2)
#    #ax.set_xlim(0.0, 0.1)
#    ax.plot(x, y - y_ref, linewidth=0.7)
#    for S in samples:
#        minima = find_minima(S)
#        ax.scatter(S[:,0], np.zeros(S.shape[0]),
#                   marker='.', zorder=2, edgecolor='none')
#        ax.scatter(S[minima, 0], np.zeros(len(minima)), color='k',
#                   marker='x', zorder=3, linewidth=0.5)
#    ax.set_ylim(ax.get_ylim())
#    mask = (x >= ax.get_xlim()[0]) & (x <= ax.get_xlim()[1])
#    ax.plot(x[mask], 1e-10*y_ref[mask], color='tab:red', linestyle='--', linewidth=0.7)
#    twax = ax.twinx()
#    twax.plot(x, (y - y_ref) / y_ref)
#    twax.set_yscale('log')
#    fig.savefig('test_barycentric_lagrange_0.pdf')


#    print("test_barycentric_lagrange_0:")
#    print("y:",y)
#    print("y2:", y_ref)
#    raise RuntimeError()


def test_barycentric_lagrange_interpolator():
    _test_barycentric_lagrange_interpolator()
