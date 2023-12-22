# Heat flow posterior predictive distribution taking into account
# spatial constraints.
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
from typing import Union
from numpy.typing import ArrayLike
from .prior import GammaConjugatePrior
from ..coverings.rdisks import bootstrap_data_selection
from .backend import gamma_conjugate_prior_predictive_common_norm,\
                     gamma_conjugate_prior_predictive_cdf_common_norm
from warnings import warn

class HeatFlowPredictive:
    """
    Posterior predictive distribution of regional heat flow taking
    into account spatial constraints (i.e. minimum distance) of the
    heat flow values.

    Parameters
    ----------
    q : array_like
        Regional distribution of :math:`N` heat flow values. Has
        to have the unit that the
        gamma conjugate prior :python:`gcp` is optimized for.
    x : array_like
        :math:`x` coordinates of the heat flow values.
    y : array_like
        :math:`y` coordinates of the heat flow values.
    gcp : reheatfunq.regional.GammaConjugatePrior
        Gamma conjugate prior.
    dmin : float, optional
        Minimum distance to be enforced between heat flow values of
        one independent sample (in m).
    n_bootstrap : int, optional
        Number of randomized selections of :python:`q` subsets
        adhering to the :math:`d_\\mathrm{min}` criterion.
    """

    def __init__(self,
                 q: ArrayLike,
                 x: ArrayLike,
                 y: ArrayLike,
                 gcp: Union[GammaConjugatePrior,tuple],
                 dmin: float = 20e3,
                 n_bootstrap: int = 1000):
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

        # Set all pre-defined attributes:
        self.q = q
        self.xy = xy
        self.gcp = gcp
        self.dmin = float(dmin)

        # Perform a bootstrap of the data under the constraint of the
        # minimum inter-data distance. The result will be a list of
        # index lists, each associated with a multiplicity that determines
        # how often the data set has been generated.
        #    bootstrap = [[m0, [i00, i01, ...]], [m1, [i10, i11, ...]], ...]
        self.bootstrap = bootstrap_data_selection(xy, self.dmin,
                                                  int(n_bootstrap))

        # Ensure that the bootstrap samples all data points:
        if set(int(s) for S in self.bootstrap for s in S[1]) \
           != set(range(q.size)):
            warn("The bootstrap did not sample all heat flow data points in "
                 "the area.")

        # Compute the Bayesian update for each bootstrap sample:
        updated = [gcp.updated(q[sample[1]]) for sample in self.bootstrap]
        self.lp = np.array([u.lp for u in updated])
        self.s = np.array([u.s for u in updated])
        self.n = np.array([u.n for u in updated])
        self.v = np.array([u.v for u in updated])


    def cdf(self, q: ArrayLike, epsabs: float = 0.0,
            epsrel: float = 1e-10):
        """
        Computes the cumulative distribution function.

        Parameters
        ----------
        q : array_like
            Heat flow :math:`q` at which to evaluate the CDF.
        epsabs : float, optional
            Absolute tolerance parameter passed to the
            quadrature engines.
        epsrel : float, optional
            Relative tolerance parameter passed to the
            quadrature engines.

        Returns
        -------
        cdf : array_like
           Posterior predictive cumulative distribution of
           regional heat flow.
        """
        # Make sure that q is C-contiguous:
        q = np.ascontiguousarray(q)
        res = np.empty_like(q)
        gamma_conjugate_prior_predictive_cdf_common_norm(q,
            self.lp, self.s, self.n, self.v, self.gcp.amin,
            epsabs, epsrel, res)

        return res


    def pdf(self, q: ArrayLike, epsabs: float = 0.0,
            epsrel: float = 1e-10):
        """
        Computes the probability distribution function.

        Parameters
        ----------
        q : array_like
            Heat flow :math:`q` at which to evaluate the CDF.
        epsabs : float, optional
            Absolute tolerance parameter passed to the
            quadrature engines.
        epsrel : float, optional
            Relative tolerance parameter passed to the
            quadrature engines.

        Returns
        -------
        pdf : array_like
           Posterior predictive probability distribution of
           regional heat flow.
        """
        # Make sure that q is C-contiguous:
        q = np.ascontiguousarray(q)
        res = np.empty_like(q)
        gamma_conjugate_prior_predictive_common_norm(q,
            self.lp, self.s, self.n, self.v, self.gcp.amin,
            epsabs, epsrel, res)

        return res
