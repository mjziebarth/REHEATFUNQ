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
from typing import Union, Literal
from numpy.typing import ArrayLike
from .anomaly import Anomaly
from ..regional import GammaConjugatePrior
from ..coverings.rdisks import bootstrap_data_selection
from .bayes import marginal_posterior_pdf_batch, marginal_posterior_cdf_batch, \
                   marginal_posterior_tail_batch, \
                   marginal_posterior_tail_quantiles_batch


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
    anomaly : reheatfunq.anomaly.Anomaly
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
                 anomaly: Anomaly,
                 gcp: Union[GammaConjugatePrior,tuple],
                 dmin: float = 20e3,
                 n_bootstrap: int = 1000,
                 heat_flow_unit: Literal['mW/m²','W/m²'] = 'mW/m²'
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

        if not isinstance(anomaly, Anomaly):
            raise TypeError("`anomaly` needs to be an Anomaly instance.")

        if heat_flow_unit not in ('mW/m²', 'W/m²'):
            raise TypeError("`heat_flow_unit` must be one of 'mW/m²' or "
                            "'W/m²'.")

        # Set all pre-defined attributes:
        self.q = q
        self.xy = xy
        self.gcp = gcp
        self.dmin = float(dmin)
        self.anomaly = anomaly
        self.heat_flow_unit = heat_flow_unit

        # Derived quantities:
        self.c = anomaly(xy)
        cmask = self.c > 0.0
        if not np.any(cmask):
            raise RuntimeError("No data point is affected by the anomaly!")

        # Handle the unit:
        if heat_flow_unit == "mW/m²":
            self.c *= 1e3

        # Compute the maximum power, defined by that heat flow
        # data point which is the first to be reduced to zero by
        # subtracting the anomaly signature:
        self.PHmax_global = np.min(self.q[cmask] / self.c[cmask])

        # Perform the bootstrap:
        self.bootstrap = bootstrap_data_selection(xy, self.dmin,
                                                  int(n_bootstrap))

        self.PHmax = max(min(self.q[i]/self.c[i] for i in sample
                             if self.c[i] > 0.0)
                         for w,sample in self.bootstrap)


    def pdf(self, P_H: ArrayLike) -> np.ndarray:
        """
        Evaluate the marginal posterior distribution in
        heat-generating power :math:`P_H`.

        Parameters
        ----------

        P_H : array_like
              The powers (in W) at which to evaluate the
              marginal posterior density.

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
        Qi = [self.q[ids] for w,ids in self.bootstrap]
        Ci = [self.c[ids] for w,ids in self.bootstrap]

        # Heavy lifting:
        return marginal_posterior_pdf_batch(P_H, p, self.gcp.s, self.gcp.n,
                                            self.gcp.v, Qi, Ci,
                                            self.gcp.amin).mean(axis=0)


    def cdf(self, P_H: ArrayLike) -> np.ndarray:
        """
        Evaluate the marginal posterior cumulative distribution
        of heat-generating power :math:`P_H`.

        Parameters
        ----------

        P_H : array_like
              The powers (in W) at which to evaluate the
              marginal posterior cumulative distribution.

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
        Qi = [self.q[ids] for w,ids in self.bootstrap]
        Ci = [self.c[ids] for w,ids in self.bootstrap]

        # Heavy lifting:
        return marginal_posterior_cdf_batch(P_H, p, self.gcp.s, self.gcp.n,
                                            self.gcp.v, Qi, Ci,
                                            self.gcp.amin)\
                   .mean(axis=0).astype(np.double)


    def tail(self, P_H: ArrayLike) -> np.ndarray:
        """
        Evaluate the posterior tail distribution (complementary
        cumulative distribution) of heat-generating power
        :math:`P_H`.

        Parameters
        ----------

        P_H : array_like
              The powers (in W) at which to evaluate the
              marginal posterior tail distribution.

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
        Qi = [self.q[ids] for w,ids in self.bootstrap]
        Ci = [self.c[ids] for w,ids in self.bootstrap]

        # Heavy lifting:
        return marginal_posterior_tail_batch(P_H, p, self.gcp.s, self.gcp.n,
                                             self.gcp.v, Qi, Ci,
                                             self.gcp.amin)\
                   .mean(axis=0).astype(np.double)


    def tail_quantiles(self, quantiles: ArrayLike,
                       dest_tol: float = 1e-3) -> np.ndarray:
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
        Qi = [self.q[ids] for w,ids in self.bootstrap]
        Ci = [self.c[ids] for w,ids in self.bootstrap]
        return marginal_posterior_tail_quantiles_batch(quantiles, p, self.gcp.s,
                                                       self.gcp.n, self.gcp.v,
                                                       Qi, Ci, self.gcp.amin,
                                                       float(dest_tol),
                                                       inplace = True)

