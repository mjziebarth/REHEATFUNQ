# Anomaly posterior using the MPMath multiprecision library.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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
from numpy.typing import NDArray
from mpmath import mp
from joblib import Memory

# Since computation of the normalization constant may take long,
# cache this computation.
# This allows us to
cache = Memory('.cache-mpm')

@cache.cache
def _init_normalization(qi: NDArray[np.float64], ci: NDArray[np.float64],
                        amin: float, p: float, s: float, n: float, v: float,
                        dps: int):
    mp.dps = dps
    N = qi.size
    assert ci.size == N
    self = AnomalyPosteriorMPM.__new__(AnomalyPosteriorMPM)
    self.qi = [mp.mpf(q) for q in qi]
    self.ci = [mp.mpf(c) for c in ci]
    self.v = v + N
    self.n = n + N
    self.s_tilde = mp.mpf(s) + sum(self.qi)
    self.B = sum(self.ci)
    self.PHmax = min(q/c for q,c in zip(self.qi, self.ci))
    self.w = self.B * self.PHmax / self.s_tilde
    self.lp_tilde = mp.log(p)
    for q in qi:
        self.lp_tilde += mp.log(q)
    self.ki = [self.PHmax * c / q for q,c in zip(self.qi, self.ci)]
    self.amin = amin

    self.imax = self.ki.index(max(self.ki))

    # Normalization:
    return self.PHmax * mp.quad(self.I, [0,1], [amin, mp.inf])


#
# The class:
#
class AnomalyPosteriorMPM:
    """
    Posterior based on MPMath.
    """
    def __init__(self, qi: NDArray[np.float64], ci: NDArray[np.float64],
                 p: float, s: float, n: float, v: float, amin: float,
                 dps: int):
        mp.dps = dps
        N = qi.size
        assert ci.size == N
        self.qi = [mp.mpf(q) for q in qi]
        self.ci = [mp.mpf(c) for c in ci]
        self.v = v + N
        self.n = n + N
        self.s_tilde = mp.mpf(s) + sum(self.qi)
        self.B = sum(self.ci)
        self.PHmax = min(q/c for q,c in zip(self.qi, self.ci))
        self.w = self.B * self.PHmax / self.s_tilde
        self.lp_tilde = mp.log(p)
        for q in qi:
            self.lp_tilde += mp.log(q)
        self.ki = [self.PHmax * c / q for q,c in zip(self.qi, self.ci)]
        self.amin = amin

        self.imax = self.ki.index(max(self.ki))

        self.Psi = _init_normalization(qi, ci, amin, p, s, n, v, dps)

        # Compute h:
        self.h = None
        self.compute_h()


    def lI(self, z, a):
        return (mp.loggamma(self.v * a) + (a - 1.0) * self.lp_tilde
                - self.n * mp.loggamma(a) - self.v * a * mp.log(self.s_tilde)
                + (a - 1) * sum(mp.log1p(-k * z) for k in self.ki)
                - self.v * a * mp.log1p(-self.w * z))

    def I(self, z, a):
        return mp.exp(self.lI(z,a))


    def pdf(self, PH, ymax=None):
        z = PH / self.PHmax
        if z >= 1.0:
            return mp.mpf(0.0)
        elif z < 0.0:
            return mp.mpf(0.0)
        if ymax is None:
            return mp.quad(lambda a : self.I(z, a) / self.Psi, [self.amin, mp.inf])
        else:
            y = 1 - z
            if y <= ymax:
                return mp.quad(lambda a : self.I2(z, a) / self.Psi, [self.amin, mp.inf])
            else:
                return mp.quad(lambda a : self.I(z, a) / self.Psi, [self.amin, mp.inf])


    def I1(self, z, a):
        y = mp.mpf(1.0) - mp.mpf(z)
        a = mp.mpf(a)
        return mp.exp(
            mp.loggamma(self.v * a) + (a - 1.0) * self.lp_tilde
            - self.n * mp.loggamma(a) - self.v * a * mp.log(self.s_tilde)
            + (a - 1.0) * mp.log(y) + mp.log(self.g1(y, a))
        )


    def g1(self, y, a):
        one = mp.mpf(1.0)
        y = mp.mpf(y)
        a = mp.mpf(a)
        nom = (a - 1.0) * sum(mp.log(one - k + k*y) for i,k in enumerate(self.ki) if i != self.imax)
        denom = self.v * a * mp.log(1.0 - self.w + self.w * y)
        return mp.exp(nom - denom)


    def compute_h(self):
        if self.h is None:
            h = [mp.mpf(0.0) for i in range(4)]
            one = mp.mpf(1.0)
            h[0] = one
            # TODO!
            for i,k in enumerate(self.ki):
                if i == self.imax:
                    continue
                for j in range(3,0,-1):
                    h[j] = (one - k) * h[j] + k * h[j-1]
                h[0] *= (1.0 - k)

            # Also save the variables:
            self.h = h

        return self.h


    def compute_C(self, a):
        a = mp.mpf(a)
        v = self.v
        w = self.w
        h0 = self.h[0]
        h1 = self.h[1]
        h2 = self.h[2]
        h3 = self.h[3]
        C0 = mp.mpf(1.0)
        C1 = (a - 1) * h1 / h0 - v * a * w / (1 - w)
        C2 = (
            (a - 1) * (a - 2) * h1**2 / h0**2 + 2*(a - 1) * h2 / h0 - 2*v*a*w*(a+1)*h1/(h0*(1-w))
            + v*a*(v*a - 1)*w**2/(1 - w)**2
        )
        C3 = (a**3*(v**3*w**3 + 3*h1/h0*v**2*w**2*(w-1) + 3*h1**2/h0**2*v*w*(w*(w-2)+1) + h1**3/h0**3*(w*(w**2 - 3*w + 3) - 1))
              + 3*a**2*(v**2*w**3 + h1/h0*v*w**2*(v-1)*(1-w) + 2*h2/h0*v*w*(w-1)**2
                        + 3*h1**2/h0**2*v*w*(w*(2-w)-1) + 2*(h1*h2/h0**2 - h1**3/h0**3)*(w*(w**2-3*w+3) - 1))
              + a * (2*v*w**3 + 3*v*w*(w*(h1/h0*(1-w) + 2*h2/h0*(2-w)) - 2*h2/h0) + 6*h1**2/h0**2*v*w*(w**2 - 2*w + 1)
                     + (6*h3/h0 - 18*h1*h2/h0**2 + 11*h1**3/h0**3) * (w**3 - 3*w**2 + 3*w - 1))
             ) / (w**3 - 3*w**2 + 3*w - 1)\
           + 6*(2*h1*h2/h0**2 - h3/h0 - h1**3/h0**3)

        return C0, C1, C2, C3


    def I2(self, z, a, ym=None, integral=False):
        y = mp.mpf(1.0) - mp.mpf(z)
        a = mp.mpf(a)
        C = self.compute_C(a)
        if integral:
            expansion = sum(ym**(a + k)/(a+k) * C[k] for k in range(4))
        else:
            expansion = sum(y**(a + k - 1) * C[k] for k in range(4))
            if expansion <= 0:
                sign = -1.0
            else:
                sign = 1.0

        return mp.exp(
            mp.loggamma(self.v * a) + (a - 1.0) * (self.lp_tilde + mp.log(self.h[0]))
            - self.n * mp.loggamma(a) - self.v * a * (mp.log(self.s_tilde) + mp.log1p(-self.w))
            + mp.log(abs(expansion))
        ) * sign
