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
from typing import Iterable, Optional, Literal
from numpy.typing import ArrayLike

import numpy as np
from math import exp, log, inf, sqrt, log10
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


    def visualize(self, ax, distributions: Optional[list[ArrayLike]] = None,
                  cax: Optional[object] = None, log_axes: bool = True,
                  cmap='inferno', color_scale: Literal['log','lin'] = 'log',
                  plot_mean: bool = True, q_mean: float = 68.3,
                  q_plot: list[tuple[float,float,float,str] | float] = [],
                  qstd_plot: list[tuple[float,float,float,str] | float] = []):
        """
        Visualize this GammaConjugatePrior instance on an axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
           The :class:`matplotlib.axes.Axes` to visualize on.
        distributions : list[array_like], optional
           A set of aggregate heat flow distributions, each given
           as a one-dimensional :class:`numpy.ndarray` of heat flow
           values in :math:`\\mathrm{mW}/\\mathrm{m}^2`. Each
           distribution will be displayed via its :math:`\\alpha`
           and :math:`\\beta` maximum likelihood estimate, indicating
           regions of interest.
           This may also determine the extent of the plot.
        cax : matplotlib.axes.Axes, optional
           The :class:`matplotlib.axes.Axes` for plotting a color bar.
        log_axes : bool, optional
           If **True**, set the axes scale to logarithmic, else use
           linear axes.
        cmap : str or matplotlib.colors.Colormap, optional
           Which color map to use for the background probability
           visualization.
        color_scale : Literal['log','lin'], optional
           If `'log'`, plot log-probability in background, else
           plot probability linearly.
        plot_mean : bool, optional
           If `'False'`, do not plot the mean heat flow lines.
        q_mean : float, optional
           The global mean heat flow in :math:`\\mathrm{mW}/\\mathrm{m}^2`.
           The default value is 68.3 from Lucazeau (2019).
        q_plot : list[tuple[float,float,float,str] | float], optional
           A list of additional average heat flow values to display.
           For each *q* a line through the :math:`(\\alpha,\\beta)` parameter
           space, enumerating parameter combinations whose distributions
           average to the given *q*. Each entry in `q_plot` needs to be
           either a float *q* or a tuple (*q*,*amin*,*amax*,*c*), where *amin*
           and *amax* denote the :math:`\\alpha`-interval within which the
           line should be plotted, and *c* is the color.
        qstd_plot : list[tuple[float,float,float,str] | float], optional
           A list of additional heat flow standard deviations to display.
           For each *qstd* a line through the :math:`(\\alpha,\\beta)` parameter
           space, enumerating parameter combinations whose distributions
           are quantified by a standard deviation *qstd*. Each entry in
           `qstd_plot` needs to be either a float *qstd* or a tuple
           (*q*,*amin*,*amax*,*c*), where *amin* and *amax* denote the
           :math:`\\alpha`-interval within which the line should be plotted,
           and *c* is the color.

        Notes
        -----
        Lucazeau, F. (2019). Analysis and mapping of an updated terrestrial
           heat flow data set. Geochemistry, Geophysics, Geosystems, 20,
           4001– 4024. https://doi.org/10.1029/2019GC008389
        """
        # Determine maximum likelihood estimates of the distributions:
        if distributions is not None:
            ai, bi = np.array([gamma_mle(dist, amin=self.amin)
                               for dist in distributions]).T
            amin = max(self.amin, ai.min()*0.8)
            amax = max(self.amin, ai.max()/0.8)
            bmin = bi.min() * 0.8
            bmax = bi.max() / 0.8
        else:
            ai = bi = None
            amin = self.amin
            amax = 1000.0
            bmin = 1e-2
            bmax = 50.0

        # Generate the coordinate grid and evaluate the prior:
        if log_axes:
            aplot = np.geomspace(amin, amax, 101)
            bplot = np.geomspace(bmin, bmax, 100)
        else:
            aplot = np.linspace(amin, amax, 101)
            bplot = np.linspace(bmin, bmax, 100)

        ag, bg = np.meshgrid(aplot, bplot)
        zg = self.log_probability(ag.flatten(), bg.flatten()).reshape(ag.shape)

        # Plot.
        # 1) The background color:
        if color_scale == 'log':
            zg *= log10(exp(1))
            vmax = zg.max()
            vmin = vmax - 9.0
        else:
            np.exp(zg, out=zg)
            vmax = zg.max()
            vmin = 0.0
        h = ax.pcolormesh(aplot, bplot, zg, cmap=cmap, vmin=vmin, vmax=vmax)

        # 2) The average heat flow isolines:
        if log_axes:
            a_plot = np.geomspace(amin / 0.9, amax * 0.9)
        else:
            a_plot = np.linspace(amin + 0.05*(amax-amin),
                                 amax - 0.05*(amax-amin))
        ax.plot(a_plot, a_plot / q_mean, color='lightskyblue', linewidth=1.0,
                zorder=1)
        ax.text(2, 4.3e-2, f'${q_mean}\,\\mathrm{{mW}}/\\mathrm{{m}}^2$',
                fontsize=8, rotation=45, color='lightskyblue')
        for entry in q_plot:
            if isinstance(entry, tuple):
                q, qamin, qamax, color = entry
            else:
                q = entry
                qamin = amin
                qamax = amax
                color = 'lightgray'
            ax.plot(a_plot[(a_plot >= qamin) & (a_plot <= qamax)],
                    a_plot[(a_plot >= qamin) & (a_plot <= qamax)] / q,
                    color=color, linewidth=0.7, zorder=1)
            ax.text(sqrt(qamin * qamax), 1.2 * sqrt(qamin * qamax) / q,
                    f'${q}\,\\mathrm{{mW}}/\\mathrm{{m}}^2$', fontsize=8,
                    rotation=45, color=color)

        # 3) The standard deviation isolines:
        for entry in qstd_plot:
            if isinstance(entry, tuple):
                qstd, qamin, qamax, color = entry
            else:
                qstd = entry
                qamin = amin
                qamax = amax
                color = 'k'
            ax.plot(a_plot[(a_plot >= qamin) & (a_plot <= qamax)],
                    np.sqrt(a_plot[(a_plot >= qamin) & (a_plot <= qamax)])
                        / qstd,
                    linestyle=':', color=color, linewidth=0.7)
            albl = sqrt(qamin * qamax)
            blbl = sqrt(albl) / qstd
            ax.text(albl, 1.1*blbl, f'${qstd}\,\\mathrm{{mW}}/\\mathrm{{m}}^2$',
                    fontsize=8, rotation=27, color=color)

        if ai is not None:
            ax.scatter(ai, bi, marker='.', color='tab:orange')
        if log_axes:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('$a$')
        ax.set_ylabel('$b$')
        ax.set_xlim(amin, amax)
        ax.set_ylim(bmin, bmax)
        cbar = ax.figure.colorbar(h, cax=cax)
        cticks = cbar.ax.get_yticks()
        cbar.ax.set_yticks(cticks)
        cbar.ax.set_yticklabels([str(10**t) for t in cticks])




    # Aliases:
    mle = maximum_likelihood_estimate





def default_prior() -> GammaConjugatePrior:
    """
    The default gamma conjugate prior from the REHEATFUNQ
    model description paper (Ziebarth *et al.*, 2022a).

    Ziebarth, M. J. and ....
    """
    return GammaConjugatePrior()
