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
                     gamma_conjugate_prior_mle, \
                     gamma_conjugate_prior_bulk_log_p, \
                     gamma_mle


class GammaConjugatePrior:
    """
    Gamma conjugate prior.
    """
    lp : float
    s : float
    n : float
    v : float

    def __init__(self, p: Optional[float], s: float, n: float, v: float,
                 lp: Optional[float] = None):
        if p is None:
            self.lp = float(lp)
        else:
            self.lp = log(float(p))
        self.s = float(s)
        self.n = float(n)
        self.v = float(v)


    def __repr__(self):
        return "GammaConjugatePrior(p=" + str(exp(self.lp)) + ", s=" \
               + str(self.s) + ", n=" + str(self.n) + ", ν=" + str(self.v) + ")"


    def updated(self, q: ArrayLike):
        """
        Perform a Bayesian update.
        """
        q = np.array(q, copy=True)
        s  = self.s + q.sum()
        n  = self.n + q.size
        v  = self.v + q.size
        lp = self.lp + np.log(q, out=q).sum()
        return GammaConjugatePrior(None, s, n, v, lp)


    def log_likelihood(self, a: ArrayLike, b: ArrayLike) -> float:
        """
        Evaluate the log-likelihood at a parameter point.
        """
        return gamma_conjugate_prior_logL(a, b, self.lp, self.s, self.n, self.v)


    def log_probability(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """
        Evaluate the logarithm of the probability at parameter points.
        """
        return gamma_conjugate_prior_bulk_log_p(a, b, self.lp, self.s, self.n,
                                                self.v)


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
                                                self.v, inplace)


    @staticmethod
    def maximum_likelihood_estimate(a: ArrayLike, b: ArrayLike,
                                    p0: float = 1.0, s0: float = 1.0,
                                    n0: float = 1.5, v0: float = 1.0,
                                    nv_surplus_min: float = 0.04,
                                    vmin: float = 0.1, epsabs: float = 0.0,
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
                                                vmin=vmin, epsabs=epsabs,
                                                epsrel=epsrel)
        return GammaConjugatePrior(None, s, n, v, lp)


    @staticmethod
    def minimum_surprise_estimate(hf_samples: list[ArrayLike],
                                  p0: float = 1.0, s0: float = 1.0,
                                  n0: float = 1.5, v0: float = 1.0,
                                  nv_surplus_min: float = 0.04,
                                  vmin: float = 0.1, epsabs: float = 0.0,
                                  epsrel: float = 1e-10,
                                  amin = 1e-3
        ) -> GammaConjugatePrior:
        """
        Compute the estimate that minimizes the maximum Kullback-Leibler
        divergence between then conjugate prior and any of the likelihoods

        """
        # Sanity:
        qset = [np.ascontiguousarray(q) for q in hf_samples]

        # Compute the extents:
        amin_ = amin
        amin = inf
        amax = -inf
        bmin = inf
        bmax = -inf
        for q in qset:
            a,b = gamma_mle(q, amin_)
            amin = min(a, amin)
            amax = max(a, amax)
            bmin = min(b, bmin)
            bmax = max(b, bmax)

        print("alim:",amin,amax)
        print("blim:",bmin,bmax)

        # Trial points:
        atr = np.linspace(0.01*amin, 10*amax, 1000)
        btr = np.linspace(0.01*bmin, 10*bmax, 1000)
        ag,bg = np.meshgrid(atr, btr)
        ag = ag.flatten()
        bg = bg.flatten()

        # Evaluate posteriors for an "uninformative" prior
        # for all the data sets:
        logP = np.empty((len(qset), ag.size))
        for i,q in enumerate(qset):
            gcp = GammaConjugatePrior(1.0, 0.0, 0.0, 0.0).updated(q)
            #print("gcp[",i,"]:",gcp)
            logP[i,:] = gcp.log_probability(ag, bg)

        # Mask irrelevant ones:
        mask = np.any(logP >= logP.max(axis=1)[:,np.newaxis] - 50, axis=0)
        print("nonzero:", np.count_nonzero(mask))
        logP = logP[:,mask]
        ag = ag[mask]
        bg = bg[mask]

        P = np.exp(logP)
        print("P:",P.sum(axis=1).min(), P.sum(axis=1).max())

        # Cost function for the optimization:
        def cost(x):
            # Retrieve parameters:
            lp = x[0]
            s = exp(x[1])
            v = x[3]
            n = v * (1.0 + x[2])

            # Compute the log probability at the integration points:
            logQ = GammaConjugatePrior(None, s, n, v, lp).log_probability(ag,bg)
            #Q = np.exp(logQ)

            # Compute the Kullback-Leibler divergences:
            #C = 0.0
            #KL = (Q * (logQ - logP)).sum(axis=1)
            KL = (P * (logP - logQ)).sum(axis=1)
            C = np.abs(KL).max()
            #for i in range(len(qset)):
            #    #KL = (P[i,:] * (logP[i,:] - logQ)).sum()
            #    KL = (Q * (logQ - logP[i,:])).sum()
            #    # C = max(KL, KLmax)
            #    C += abs(KL)

            # Return the maximum expected surprise:
            print("  ",C)
            return C

        res = minimize(cost, (log(p0), log(s0), max(n0/v0-1.0, nv_surplus_min),
                              v0),
                       bounds=((-inf, inf), (-inf, inf), (nv_surplus_min, inf),
                               (vmin, inf)),
                       method='Nelder-Mead')

        print("result:")
        print(res)

        lp = res.x[0]
        s = exp(res.x[1])
        v = res.x[3]
        n = v * (1.0 + res.x[2])
        return GammaConjugatePrior(None, s, n, v, lp)


    @staticmethod
    def equal_probability_least_squares(a: ArrayLike, b: ArrayLike,
                                        p0: float = 1.0, s0: float = 1.0,
                                        n0: float = 1.5, v0: float = 1.0,
                                        nv_surplus_min: float = 1e-3,
                                        vmin: float = 0.2, epsabs: float = 0.0,
                                        epsrel: float = 1e-10
        ) -> GammaConjugatePrior:
        """
        Compute an estimate
        """
        def cost(x):
            lp = x[0]
            s = exp(x[1])
            v = x[3]
            n = v * (1.0 + x[2])
            try:
                log_p = gamma_conjugate_prior_bulk_log_p(a, b, lp, s, n, v)
            except RuntimeError:
                return inf
            log_p_mean = log_p.mean()
            log_p -= log_p_mean
            log_p **= 2
            #log_p **= 2
            #return 20 * log_p.mean() - log_p_mean
            return 20 * log_p.max() - log_p_mean

        res = minimize(cost, (log(p0), log(s0), max(n0/v0-1.0, nv_surplus_min),
                              v0),
                       bounds=((-inf, inf), (-inf, inf), (nv_surplus_min, inf),
                               (vmin, inf)),
                       method='Nelder-Mead')
        print("result:")
        print(res)

        def cost_post(x):
            lp = x[0]
            s = exp(x[1])
            v = x[3]
            n = v * (1.0 + x[2])
            try:
                log_p = gamma_conjugate_prior_bulk_log_p(a, b, lp, s, n, v)
            except RuntimeError:
                return inf
            log_p_mean = log_p.mean()
            log_p -= log_p_mean
            #log_p **= 2
            #return log_p.mean()
            np.abs(log_p, out=log_p)
            return log_p.max()

        res = minimize(cost_post, res.x,
                       bounds=((-inf, inf), (-inf, inf), (nv_surplus_min, inf),
                               (vmin, inf)),
                       method='Nelder-Mead')

        print("result:")
        print(res)

        lp = res.x[0]
        s = exp(res.x[1])
        v = res.x[3]
        n = v * (1.0 + res.x[2])
        return GammaConjugatePrior(None, s, n, v, lp)



    # Aliases:
    mle = maximum_likelihood_estimate





def default_prior() -> GammaConjugatePrior:
    """
    The default gamma conjugate prior from the REHEATFUNQ
    model description paper (Ziebarth *et al.*, 2022a).

    Ziebarth, M. J. and ....
    """
    return GammaConjugatePrior()
