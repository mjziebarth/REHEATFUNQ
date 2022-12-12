# Test the numerics of the heat flow anomaly quantification posterior.
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
from reheatfunq.regional.backend import gamma_mle
from reheatfunq import HeatFlowAnomalyPosterior, AnomalyLS1980, \
                       GammaConjugatePrior
from reheatfunq.regional import default_prior
from reheatfunq.anomaly.bayes import marginal_posterior_pdf


def test_across_precisions_N50():
    """
    Use different numerical precisions to cross-validate the
    numerics.
    """
    AnoPH = 100e6
    N = 50

    alpha = 40.
    beta = 0.4

    ano = AnomalyLS1980(np.array([(0, -80e3), (0, 80e3)]), 14e3)

    rng = np.random.default_rng(29189)
    Q = np.sort(rng.gamma(alpha, size=N) / beta)
    x = 160e3 * (rng.random(size=N) - 0.5)
    y = 160e3 * (rng.random(size=N) - 0.5)
    xy = np.zeros((x.size,2))
    xy[:,0] = x
    xy[:,1] = y
    q = Q + 1e3*ano(xy)

    gcp = default_prior()

    hfp = HeatFlowAnomalyPosterior(q, x, y, ano, gcp, 5.0e3)

    P_H = np.linspace(0, hfp.PHmax_global, 200)

    pdf = []
    cdf = []
    tail = []
    for wp in ['double','long double','dec50']:
        pdf.append(hfp.pdf(P_H, working_precision=wp))
        cdf.append(hfp.cdf(P_H, working_precision=wp))
        tail.append(hfp.tail(P_H, working_precision=wp))

    np.testing.assert_allclose(pdf[0], pdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(pdf[1], pdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cdf[0], cdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cdf[1], cdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[0], tail[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[1], tail[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[2], 1.0 - cdf[2], rtol=0.0, atol=5e-16)


def test_across_precisions_N100():
    """
    Use different numerical precisions to cross-validate the
    numerics.
    """
    AnoPH = 100e6
    N = 100

    alpha = 40.
    beta = 0.4

    ano = AnomalyLS1980(np.array([(0, -80e3), (0, 80e3)]), 14e3)

    rng = np.random.default_rng(29189)
    Q = np.sort(rng.gamma(alpha, size=N) / beta)
    x = 160e3 * (rng.random(size=N) - 0.5)
    y = 160e3 * (rng.random(size=N) - 0.5)
    xy = np.zeros((x.size,2))
    xy[:,0] = x
    xy[:,1] = y
    q = Q + 1e3*ano(xy)

    gcp = default_prior()

    hfp = HeatFlowAnomalyPosterior(q, x, y, ano, gcp, 0.0e3)

    P_H = np.linspace(0, hfp.PHmax_global, 200)

    pdf = []
    cdf = []
    tail = []
    for wp in ['double','long double','dec50']:
        pdf.append(hfp.pdf(P_H, working_precision=wp))
        cdf.append(hfp.cdf(P_H, working_precision=wp))
        tail.append(hfp.tail(P_H, working_precision=wp))

    np.testing.assert_allclose(pdf[0],pdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(pdf[1],pdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cdf[0],cdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cdf[1],cdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[0],tail[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[1],tail[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[2], 1.0 - cdf[2], rtol=0.0, atol=1e-15)


def test_across_precisions_N1000():
    """
    Use different numerical precisions to cross-validate the
    numerics.
    """
    AnoPH = 100e6
    N = 1000

    alpha = 120.
    beta = 4.

    ano = AnomalyLS1980(np.array([(0, -80e3), (0, 80e3)]), 14e3)

    rng = np.random.default_rng(294982299)
    Q = np.sort(rng.gamma(alpha, size=N) / beta)
    x = 160e3 * (rng.random(size=N) - 0.5)
    y = 160e3 * (rng.random(size=N) - 0.5)
    xy = np.zeros((x.size,2))
    xy[:,0] = x
    xy[:,1] = y
    q = Q + 1e3*ano(xy)

    gcp = default_prior()

    hfp = HeatFlowAnomalyPosterior(q, x, y, ano, gcp, 0.0e3)

    P_H = np.linspace(0, hfp.PHmax_global, 200)

    pdf = []
    cdf = []
    tail = []
    for wp in ['double','long double','dec50']:
        pdf.append(hfp.pdf(P_H, working_precision=wp))
        cdf.append(hfp.cdf(P_H, working_precision=wp))
        tail.append(hfp.tail(P_H, working_precision=wp))

    np.testing.assert_allclose(pdf[0],pdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(pdf[1],pdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cdf[0],cdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cdf[1],cdf[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[0],tail[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[1],tail[2], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(tail[2], 1.0 - cdf[2], rtol=0.0, atol=2e-16)


def test_absolute_values():
    """
    Test the numerics against some independent values computed with
    MPMath quadrature at dps=50.
    """
    # The test configuration:
    p = 2.5220172925228335
    s = 15.373016724405591
    n = 0.2184765374862465
    v = 0.2184765374862465
    q = [ 75.70255678,  76.18744121,  79.01339271,  89.41498501, 99.3258167,
         100.00115816, 103.74341636, 106.24710206, 111.40302288, 126.80598511]
    c = [3.74039144e-12, 8.63295937e-12, 3.28571247e-12, 6.02724571e-11,
         3.95429084e-12, 2.38382437e-11, 4.73856217e-12, 1.22700220e-10,
         4.76447511e-11, 1.43570283e-11]

    # Test evaluation at:
    PH = [0.0,
          36079499986.49826,
          72158999972.99652,
          108238499959.49478,
          144317999945.99304,
          180397499932.4913,
          216476999918.98956,
          252556499905.48782,
          288635999891.9861,
          324715499878.4844,
          360794999864.9826,
          396874499851.48083,
          432953999837.9791,
          469033499824.4774,
          505112999810.97565,
          541192499797.4739,
          577271999783.9722,
          613351499770.4705,
          649430999756.9688,
          685510499743.4669,
          721589999729.9652,
          757669499716.4635,
          793748999702.9617,
          829828499689.46,
          865907999675.9583]

    # The example results, computed with (naive) MPMath integration:
    results = [1.7241192897370277e-12,
               2.2866977442930377e-12,
               2.86091040565077e-12,
               3.338331735964264e-12,
               3.596531876975022e-12,
               3.5496478248777957e-12,
               3.1952036246173426e-12,
               2.6217208227599387e-12,
               1.967463772125419e-12,
               1.3594235527807356e-12,
               8.726156417316245e-13,
               5.256399224439391e-13,
               3.001457222977609e-13,
               1.639673040751133e-13,
               8.636160270923706e-14,
               4.4112650124608125e-14,
               2.1932762532396427e-14,
               1.0628488370765047e-14,
               5.012061349931154e-15,
               2.28775249245078e-15,
               9.995794781506387e-16,
               4.090459293003127e-16,
               1.4962216749276923e-16,
               4.289599271464578e-17,
               0.0]

    # Now validate the REHEATFUNQ code:
    pdf_compare = marginal_posterior_pdf(np.array(PH), p, s, n, v, np.array(q),
                                         np.array(c), amin = 1.0,
                                         dest_tol=1e-15,
                                         working_precision = 'long double')

    np.testing.assert_allclose(pdf_compare, results, rtol=1e-11)
