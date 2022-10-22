# Gamma conjugate prior routines.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Malte J. Ziebarth
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

# Typing:
from __future__ import annotations
from typing import Iterable, Optional
from numpy.typing import ArrayLike

import numpy as np
from math import exp, log, inf, sqrt
from scipy.special import loggamma
from scipy.optimize import minimize
from .backend import gamma_conjugate_prior_logL, \
                     gamma_conjugate_prior_predictive, \
                     gamma_conjugate_prior_predictive_cdf, \
                     gamma_conjugate_prior_mle, \
                     gamma_conjugate_prior_bulk_log_p, \
                     gamma_mle, \
                     gamma_conjugate_prior_kullback_leibler


class GammaConjugatePrior:
    """
    Gamma conjugate prior.
    """
    lp : float
    s : float
    n : float
    v : float
    amin : float

    def __init__(self, p: Optional[float], s: float, n: float, v: float,
                 lp: Optional[float] = None, amin: float = 1.0):
        if p is None:
            self.lp = float(lp)
        else:
            self.lp = log(float(p))
        self.s = float(s)
        self.n = float(n)
        self.v = float(v)
        self.amin = float(amin)


    def __repr__(self):
        return "GammaConjugatePrior(p=" + str(exp(self.lp)) + ", s=" \
               + str(self.s) + ", n=" + str(self.n) + ", ν=" + str(self.v) \
               + ", amin=" + str(self.amin) + ")"


    def updated(self, q: ArrayLike):
        """
        Perform a Bayesian update.
        """
        q = np.array(q, copy=True)
        s  = self.s + q.sum()
        n  = self.n + q.size
        v  = self.v + q.size
        lp = self.lp + np.log(q, out=q).sum()
        return GammaConjugatePrior(None, s, n, v, lp, self.amin)


    def log_likelihood(self, a: ArrayLike, b: ArrayLike) -> float:
        """
        Evaluate the log-likelihood at a parameter point.
        """
        return gamma_conjugate_prior_logL(a, b, self.lp, self.s, self.n, self.v,
                                          self.amin)


    def log_probability(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """
        Evaluate the logarithm of the probability at parameter points.
        """
        return gamma_conjugate_prior_bulk_log_p(a, b, self.lp, self.s, self.n,
                                                self.v, self.amin)


    def probability(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """
        Evaluate the probability at parameter points.
        """
        log_p = self.log_probability(a, b)
        return np.exp(log_p, out=log_p)


    def posterior_predictive(self, q: ArrayLike,
                             inplace: bool = False) -> ArrayLike:
        """
        Evaluate the posterior predictive distribution for given heat
        flow `q`.
        """
        return gamma_conjugate_prior_predictive(q, self.lp, self.s, self.n,
                                                self.v, self.amin, inplace)


    def posterior_predictive_cdf(self, q: ArrayLike,
                                 inplace: bool = False) -> ArrayLike:
        """
        Evaluate the posterior predictive distribution for given heat
        flow `q`.
        """
        q = np.ascontiguousarray(q)
        return gamma_conjugate_prior_predictive_cdf(q, self.lp, self.s, self.n,
                                                    self.v, self.amin, inplace)


    def kullback_leibler(self, other: GammaConjugatePrior) -> float:
        """
        Compute the Kullback-Leibler divergence to another gamma
        conjugate prior.
        """
        return gamma_conjugate_prior_kullback_leibler(other.lp, other.s,
                                                      other.n, other.v,
                                                      self.lp, self.s, self.n,
                                                      self.v)


    @staticmethod
    def maximum_likelihood_estimate(a: ArrayLike, b: ArrayLike,
                                    p0: float = 1.0, s0: float = 1.0,
                                    n0: float = 1.5, v0: float = 1.0,
                                    nv_surplus_min: float = 0.04,
                                    vmin: float = 0.1, amin: float = 1.0,
                                    epsabs: float = 0.0,
                                    epsrel: float = 1e-10
        ) -> GammaConjugatePrior:
        """
        Compute the maximum likelihood estimate.
        """
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        lp, s, n, v = gamma_conjugate_prior_mle(a, b, p0=p0, s0=s0, n0=n0,
                                                v0=v0,
                                                nv_surplus_min=nv_surplus_min,
                                                vmin=vmin, amin=amin,
                                                epsabs=epsabs, epsrel=epsrel)
        return GammaConjugatePrior(None, s, n, v, lp, amin)


    @staticmethod
    def minimum_surprise_estimate(hf_samples: list[ArrayLike],
                                  p0: float = 1.0, s0: float = 1.0,
                                  n0: float = 1.5, v0: float = 1.0,
                                  nv_surplus_min: float = 0.04,
                                  vmin: float = 0.1, amin: float = 1.0,
                                  epsabs: float = 0.0, epsrel: float = 1e-10
        ) -> GammaConjugatePrior:
        """
        Compute the estimate that minimizes the maximum Kullback-Leibler
        divergence between then conjugate prior and any of the likelihoods

        """
        # Sanity:
        qset = [np.ascontiguousarray(q) for q in hf_samples]

        # Compute the "uninformed" estimates for each sample:
        GCP_i = [GammaConjugatePrior(1.0, 0.0, 0.0, 0.0, amin).updated(q)
                 for q in qset]

        # Cost function for the optimization:
        def cost(x):
            # Retrieve parameters:
            lp = x[0]
            s = exp(x[1])
            v = x[3]
            n = v * (1.0 + x[2])
            gcp_test = GammaConjugatePrior(None, s, n, v, lp, amin)

            # Compute the Kullback-Leibler divergences:
            KLmax = -inf
            for gcp in GCP_i:
                KL = gcp_test.kullback_leibler(gcp)
                KLmax = max(KLmax, KL)

            return KLmax

        res = minimize(cost, (log(p0), log(s0), max(n0/v0-1.0, nv_surplus_min),
                              v0),
                       bounds=((-inf, inf), (-inf, inf), (nv_surplus_min, inf),
                               (vmin, inf)),
                       method='Nelder-Mead')

        lp = res.x[0]
        s = exp(res.x[1])
        v = res.x[3]
        n = v * (1.0 + res.x[2])
        return GammaConjugatePrior(None, s, n, v, lp, amin)


    # Aliases:
    mle = maximum_likelihood_estimate





def default_prior() -> GammaConjugatePrior:
    """
    The default gamma conjugate prior from the REHEATFUNQ
    model description paper (Ziebarth *et al.*, 2022a).

    Ziebarth, M. J. and ....
    """
    return GammaConjugatePrior()
