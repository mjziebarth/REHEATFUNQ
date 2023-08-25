# Anomaly probability classification.
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

import numpy as np
from math import exp
from typing import Union, Literal, Iterable
from numpy.typing import ArrayLike
from .anomaly import Anomaly
from ..regional import GammaConjugatePrior
from ..coverings.rdisks import bootstrap_data_selection, \
                               all_restricted_samples, \
                               samples_from_discrete_distribution
from .bayes import marginal_posterior_pdf_batch, marginal_posterior_cdf_batch, \
                   marginal_posterior_tail_batch, \
                   marginal_posterior_tail_quantiles_batch, \
                   _support_float128, _support_dec50, _support_dec100


# Define the capabilities of the numerical backend:
_num_prec = ['auto','double','long double']
if _support_float128():
    _num_prec.append('float128')
if _support_dec50():
    _num_prec.append('dec50')
if _support_dec100():
    _num_prec.append('dec100')



class HeatFlowAnomalyPosterior:
    """
    This class evaluates the posterior probability of the
    strength of a heat flow anomaly, expressed by the frictional
    power :math:`P_H` on the fault, using the REHEATFUNQ model
    of regional heat flow and a set of regional heat flow data.

    Parameters
    ----------
    q : array_like
        The heat flow data of shape :python:`(N,)`.
    x : array_like
        The :math:`x` locations of the heat flow data.
        Also shape :python:`(N,)`.
    y : array_like
        The :math:`y` locations of the heat flow data.
        Also shape :python:`(N,)`.
    anomaly : reheatfunq.anomaly.Anomaly | list[reheatfunq.anomaly.Anomaly] \
              | list[tuple[float,reheatfunq.anomaly.Anomaly]]
        The model of the heat flow anomaly that can be evaluated at
        the data locations.
    gcp : reheatfunq.regional.GammaConjugatePrior
        The prior for the regional aggregate heat flow distribution.
    dmin : float
        The minimum distance between data points (in m). If data
        points closer than this distance exist in the heat flow
        data, they are not considered independent and are alternated
        in the bootstrap.
    n_bootstrap : int
        The number of permuted heat flow data sets to generate.
        If no pair of data points is closer than the minimum
        distance :math:`d_\mathrm{min}`, this parameter has no
        effect.
    heat_flow_unit : 'mW/m²' | 'W/m²'
        The unit in which the heat flow data :python:`q` are given.
    """
    def __init__(self,
                 q: ArrayLike,
                 x: ArrayLike,
                 y: ArrayLike,
                 anomaly: Anomaly | Iterable[Anomaly] |
                          Iterable[tuple[float,Anomaly]],
                 gcp: Union[GammaConjugatePrior,tuple],
                 dmin: float = 20e3,
                 n_bootstrap: int = 1000,
                 heat_flow_unit: Literal['mW/m²','W/m²'] = 'mW/m²',
                 rng: int | np.random.Generator = 127
        ):
        # Ensure that heat flow is contiguous:
        q = np.ascontiguousarray(np.array(q,copy=False).reshape(-1))

        # Merge x and y coordinates into a contiguous (N,2) array
        # and ensure that it fits to q:
        try:
            xy = np.stack((x,y), axis=1)
            assert xy.shape[1] == 2
            assert xy.ndim == 2
            assert xy.shape[0] == q.size
        except:
            raise TypeError("`x` and `y` have to be equal-sized "
                            "one-dimensional coordinate arrays that fit "
                            "to `q` in shape.")

        if not isinstance(gcp, GammaConjugatePrior):
            try:
                gcp = GammaConjugatePrior(gcp)
            except:
                raise TypeError("`gcp` needs to be a GammaConjugatePrior "
                                "instance or a tuple (p,s,n,ν).")

        #
        # The type of 'anomaly' should be a list of tuples (w_i, a_i),
        # where w_i is a weight (float) and a_i is an anomaly. This can
        # cover uncertainty in the heat transport by providing a couple
        # of heat transport solutions as Anomaly instances, and weighting
        # them, if wanted, by probability mass.
        # If not all of this information is given, fill up the remaining
        # parts sensibly.
        #
        if isinstance(anomaly, Anomaly):
            # Single anomaly, 100% weight:
            anomaly = [(1.0, anomaly)]
        else:
            # An iterable of anomalies. Convert to list so as not to lose
            # iterability:
            anomaly = list(anomaly)

            # A function that tests whether a list of non-anomlies conforms
            # to the (float,Anomaly) tuple type:
            def test_tuple(a) -> bool:
                if not isinstance(a,tuple):
                    return False
                if len(a) != 2:
                    return False
                try:
                    assert float(a[0]) > 0.0
                except:
                    return False
                return isinstance(a[1],Anomaly)

            if all(isinstance(a, Anomaly) for a in anomaly):
                # List of N anomalies, weight 1/N
                anomaly = [(1.0/len(anomaly), a) for a in anomaly]
            elif not all(test_tuple(a) for a in anomaly):
                raise TypeError("`anomaly` needs to be either an Anomaly "
                                "instance, a list of Anomaly instances, or "
                                "a list of (float,Anomaly) tuples.")
            else:
                W = sum(float(a[0]) for a in anomaly)
                if W <= 0.0:
                    raise ValueError("All anomaly weights must be positive.")
                anomaly = [(float(a[0]) / W, a[1]) for a in anomaly]

        if heat_flow_unit not in ('mW/m²', 'W/m²'):
            raise TypeError("`heat_flow_unit` must be one of 'mW/m²' or "
                            "'W/m²'.")

        n_boostrap = int(n_bootstrap)
        if n_bootstrap <= 0:
            raise ValueError("'n_bootstrap' must be a positive integer.")

        # Set all pre-defined attributes:
        self.q = q
        self.xy = xy
        self.gcp = gcp
        self.dmin = float(dmin)
        self.anomaly = anomaly
        self.heat_flow_unit = heat_flow_unit

        # Derived quantities:
        self.c = np.array([a[1](xy) for a in anomaly])
        cmask = self.c > 0.0
        if not np.any(cmask):
            raise RuntimeError("No data point is affected by the anomaly!")

        # Handle the unit:
        if heat_flow_unit == "mW/m²":
            self.c *= 1e3

        # Compute the maximum power, defined by that heat flow
        # data point which is the first to be reduced to zero by
        # subtracting the anomaly signature.
        # Note: This differs from self.PHmax only if the bootstrap does
        #       not cover all combinations of q and c - mostly unlikely.
        Nano = self.c.shape[0]
        self.PHmax_global = np.min(self.q[np.newaxis,:].repeat(Nano,0)[cmask]
                                   / self.c[cmask])

        # Perform the bootstrap:
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError("`rng` must be an integer or a numpy.random."
                            "Generator instance.")

        # The bootstrap is a uniform random sampling within the parameter
        # space:
        bootstrap_samples = \
           all_restricted_samples(xy, self.dmin, n_bootstrap, n_bootstrap, rng)

        # Now we might have two situations:
        # 1) the number of elements of 'bootstrap_samples' is less than
        #    n_bootstrap. This can happen when the permutation space is
        #    exhausted (at a lower number of conflicting node pairs,
        #    presumably). We then need to sample 'n_bootstrap' elements
        #    from this distribution according to the probabilities
        #    provided. (If we need a bootstrap, that is)
        # 2) the number of elements in 'bootstrap_samples' is equal to
        #    n_bootstrap. We can simply use the provided samples.
        w_ars = np.array([bs[0] for bs in bootstrap_samples])
        if len(bootstrap_samples) < n_bootstrap and self.c.shape[0] > 1:
            # Case 1)
            bootstrap_i = samples_from_discrete_distribution(w_ars, n_bootstrap,
                                                             rng)
            w_ars = w_ars[bootstrap_i]
            bootstrap_i = [bootstrap_samples[i][1] for i in bootstrap_i]
        else:
            bootstrap_i = [bs[1] for bs in bootstrap_samples]

        # Same for anomaly:
        if self.c.shape[0] > 1:
            bootstrap_j = [int(j) for j in rng.integers(0, self.c.shape[0],
                                                        size=n_bootstrap)]
        else:
            bootstrap_j = [0 for j in range(self.c.shape[0])]

        # Weight is the joint probability. Need to normalize these
        # discrete weights.
        bootstrap_w = [anomaly[j][0] * w_ars[j] for j in bootstrap_j]
        W = sum(bootstrap_w)
        bootstrap_w = [float(w / W) for w in bootstrap_w]

        self.bootstrap = list(zip(bootstrap_w, bootstrap_j, bootstrap_i))

        self.PHmax = max(min(self.q[i]/self.c[j,i] for i in sample
                             if self.c[j,i] > 0.0)
                         for w,j,sample in self.bootstrap)

        self.PHmax_global = max(min(self.q[i]/self.c[j,i]
                                    for i in range(len(self.q))
                                    if self.c[j,i] > 0.0)
                                for j in range(self.c.shape[0]))


    def pdf(self, P_H: ArrayLike, dest_tol: float = 1e-8,
            working_precision: Literal[_num_prec] = 'auto') -> np.ndarray:
        """
        Evaluate the marginal posterior distribution in
        heat-generating power :math:`P_H`.

        Parameters
        ----------
        P_H : array_like
              The powers (in W) at which to evaluate the
              marginal posterior density.
        dest_tol : float, optional
              The destination tolerance to use for the power
              :math:`P_H` integration.
        working_precision : 'auto' | 'double' | 'long double', optional
              The precision of the internal numerical computations.
              The higher the precision, the more likely it is to
              obtain a precise result for large data sets. The
              trade off is a longer run time.
              If the respective flags have been set at compile
              time, additional options 'float128' (GCC 128bit
              floating point), 'dec50' (boost 50-digit multiprecision),
              and 'dec100' (boost 100-digit multiprecision) are
              available.

        Returns
        -------
        pdf : array_like
              The marginal posterior probability density of
              heat-generating power :math:`P_H` evaluated at
              the given :python:`P_H`.
        """
        # Make sure that P_H is C-contiguous:
        P_H = np.ascontiguousarray(P_H)

        # Prepare:
        p = exp(self.gcp.lp)
        Qi = [self.q[ids] for w,j,ids in self.bootstrap]
        Ci = [self.c[j,ids] for w,j,ids in self.bootstrap]

        # Heavy lifting:
        return marginal_posterior_pdf_batch(P_H, p, self.gcp.s, self.gcp.n,
                                            self.gcp.v, Qi, Ci,
                                            self.gcp.amin).mean(axis=0)


    def cdf(self, P_H: ArrayLike, dest_tol: float = 1e-8,
            working_precision: Literal[_num_prec] = 'auto') -> np.ndarray:
        """
        Evaluate the marginal posterior cumulative distribution
        of heat-generating power :math:`P_H`.

        Parameters
        ----------
        P_H : array_like
              The powers (in W) at which to evaluate the
              marginal posterior cumulative distribution.
        dest_tol : float, optional
              The destination tolerance to use for the power
              :math:`P_H` integration.
        working_precision : 'auto' | 'double' | 'long double', optional
              The precision of the internal numerical computations.
              The higher the precision, the more likely it is to
              obtain a precise result for large data sets. The
              trade off is a longer run time.
              If the respective flags have been set at compile
              time, additional options 'float128' (GCC 128bit
              floating point), 'dec50' (boost 50-digit multiprecision),
              and 'dec100' (boost 100-digit multiprecision) are
              available.

        Returns
        -------
        cdf : array_like
              The marginal posterior cumulative distribution
              of heat-generating power :math:`P_H` evaluated
              at the given :python:`P_H`.
        """
        # Make sure that P_H is C-contiguous:
        P_H = np.ascontiguousarray(P_H)

        # Prepare:
        p = exp(self.gcp.lp)
        Qi = [self.q[ids] for w,j,ids in self.bootstrap]
        Ci = [self.c[j,ids] for w,j,ids in self.bootstrap]

        # Heavy lifting:
        return marginal_posterior_cdf_batch(P_H, p, self.gcp.s, self.gcp.n,
                                            self.gcp.v, Qi, Ci,
                                            self.gcp.amin)\
                   .mean(axis=0).astype(np.double)


    def tail(self, P_H: ArrayLike, dest_tol: float = 1e-8,
             working_precision: Literal[_num_prec] = 'auto') -> np.ndarray:
        """
        Evaluate the posterior tail distribution (complementary
        cumulative distribution) of heat-generating power
        :math:`P_H`.

        Parameters
        ----------
        P_H : array_like
              The powers (in W) at which to evaluate the
              marginal posterior tail distribution.
        dest_tol : float, optional
              The destination tolerance to use for the power
              :math:`P_H` integration.
        working_precision : 'auto' | 'double' | 'long double', optional
              The precision of the internal numerical computations.
              The higher the precision, the more likely it is to
              obtain a precise result for large data sets. The
              trade off is a longer run time.
              If the respective flags have been set at compile
              time, additional options 'float128' (GCC 128bit
              floating point), 'dec50' (boost 50-digit multiprecision),
              and 'dec100' (boost 100-digit multiprecision) are
              available.

        Returns
        -------
        tail : array_like
              The marginal posterior tail distribution
              of heat-generating power :math:`P_H` evaluated
              at the given :code:`P_H`.
        """
        # Make sure that P_H is C-contiguous:
        P_H = np.ascontiguousarray(P_H)

        # Prepare:
        p = exp(self.gcp.lp)
        Qi = [self.q[ids] for w,j,ids in self.bootstrap]
        Ci = [self.c[j,ids] for w,j,ids in self.bootstrap]

        # Heavy lifting:
        return marginal_posterior_tail_batch(P_H, p, self.gcp.s, self.gcp.n,
                                             self.gcp.v, Qi, Ci,
                                             self.gcp.amin)\
                   .mean(axis=0).astype(np.double)


    def tail_quantiles(self, quantiles: ArrayLike,
                       dest_tol: float = 1e-8,
                       working_precision: Literal[_num_prec] = 'auto',
                       method: Literal['bli','1.3.3'] = 'bli',
                       n_chebyshev: int = 100) -> np.ndarray:
        """
        Compute posterior tail quantiles, that is, heat-generating
        powers :math:`P_H` at which the complementary cumulative
        distribution of :math:`P_H` has fallen to level :math:`x`.

        Parameters
        ----------
        quantiles : array_like
            The tail quantiles to compute.
        dest_tol : float, optional
            The tolerance to which to compute the powers
            :math:`P_H`.
        working_precision : 'auto' | 'double' | 'long double', optional
            The precision of the internal numerical computations.
            The higher the precision, the more likely it is to
            obtain a precise result for large data sets. The
            trade off is a longer run time.
            If the respective flags have been set at compile
            time, additional options 'float128' (GCC 128bit
            floating point), 'dec50' (boost 50-digit multiprecision),
            and 'dec100' (boost 100-digit multiprecision) are
            available.
            Note: currently only used if :code:`method == 'bli'`.
        method : 'bli' | 'old', optional
            Defines which method to use for inverting the tail
            distribution for the tail quantile.
              * 'bli' : Use barycentric Lagrange interpolation on a
                number of tail distribution evaluations.
              * '1.3.3' : An explicit method based on simultaneous PDF
                integration and bisection. Default method of versions
                1.3.3 and before.
        n_chebyshev : int, optional
            Number of Chebyshev points to evaluate the tail
            distribution at if the method is barycentric Lagrange
            interpolation.

        Returns
        -------
        P_H : array_like
              The heat-generating power :math:`P_H` at which the
              posterior tail distribution evaluates to :code:`x`.
        """
        # Make sure that quantiles is C-contiguous:
        quantiles = np.array(quantiles, float, copy=True, order='C', ndmin=1)

        # Prepare:
        p = exp(self.gcp.lp)
        Qi = [self.q[ids] for w,j,ids in self.bootstrap]
        Ci = [self.c[j,ids] for w,j,ids in self.bootstrap]
        return marginal_posterior_tail_quantiles_batch(quantiles, p, self.gcp.s,
                                                       self.gcp.n, self.gcp.v,
                                                       Qi, Ci, self.gcp.amin,
                                                       float(dest_tol), method,
                                                       working_precision,
                                                       n_chebyshev,
                                                       inplace = True)

