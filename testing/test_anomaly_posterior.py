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
from reheatfunq.anomaly.postbackend import CppAnomalyPosterior
from reheatfunq.anomaly.posterior import support_float128,\
        support_dec50, support_dec100
from warnings import warn

#double:            2.2204460492503130808e-16
#long double:       1.084202172485504434e-19
#cpp_dec_float128:  1.9259299443872358531e-34
#cpp_dec_float_50:  1e-49
#cpp_dec_float_100: 9.9999999999999999997e-100

_PREC_RTOL_FLOAT128 = np.sqrt(1.926e-34)
_PREC_RTOL_DEC50 = np.sqrt(1e-49)
_PREC_RTOL_DEC100 = np.sqrt(1e-99)

_PREC_RTOL = [np.sqrt(2.2e-16), np.sqrt(1.08e-19)]

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
    q = Q + 1e3*AnoPH*ano(xy)

    gcp = default_prior()

    hfps = []
    for prec in ['double','long double']:
        print("prec:",prec)
        hfps.append(
            HeatFlowAnomalyPosterior(q, x, y, ano, gcp, 5.0e3,
                                     precision=prec)
        )

    P_H = np.linspace(0, max(hfp.PHmax_global for hfp in hfps), 200)

    pdf = []
    cdf = []
    tail = []
    for hfp in hfps:
        pdf.append(hfp.pdf(P_H))
        cdf.append(hfp.cdf(P_H))
        tail.append(hfp.tail(P_H))

    for i in range(len(pdf)-1):
        np.testing.assert_allclose(pdf[i], pdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(cdf[i], cdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(tail[i], tail[-1], rtol=_PREC_RTOL[i], atol=0.0)

    np.testing.assert_allclose(tail[-1], 1.0 - cdf[-1], rtol=0.0, atol=5e-16)


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
    q = Q + 1e3*AnoPH*ano(xy)

    gcp = default_prior()

    hfps = []
    for prec in ['double','long double']:
        print("prec:",prec)
        hfps.append(
            HeatFlowAnomalyPosterior(q, x, y, ano, gcp, 5.0e3,
                                     precision=prec)
        )

    P_H = np.linspace(0, max(hfp.PHmax_global for hfp in hfps), 200)

    pdf = []
    cdf = []
    tail = []
    for hfp in hfps:
        pdf.append(hfp.pdf(P_H))
        cdf.append(hfp.cdf(P_H))
        tail.append(hfp.tail(P_H))


    for i in range(len(pdf)-1):
        np.testing.assert_allclose(pdf[i], pdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(cdf[i], cdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(tail[i], tail[-1], rtol=_PREC_RTOL[i], atol=0.0)

    np.testing.assert_allclose(tail[-1], 1.0 - cdf[-1], rtol=0.0, atol=5e-16)


def test_across_precisions_N1000():
    """
    Use different numerical precisions to cross-validate the
    numerics.
    """
    print("test_across_precisions_N1000")
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
    q = Q + 1e3*AnoPH*ano(xy)

    gcp = default_prior()

    hfps = []
    for prec in ['double','long double']:
        print("prec:",prec)
        try:
            hfps.append(
                HeatFlowAnomalyPosterior(q, x, y, ano, gcp, 5.0e3,
                                        precision=prec,
                                        bli_max_refinements=5)
            )
        except Exception as e:
            from pickle import Pickler
            with open('dump.pickle','wb') as f:
                Pickler(f).dump((q, x, y))
            raise e

    P_H = np.linspace(0, max(hfp.PHmax_global for hfp in hfps), 200)
    P_H = np.linspace(0, max(hfp.PHmax_global for hfp in hfps), 20)

    pdf = []
    cdf = []
    tail = []
    for hfp in hfps:
        pdf.append(hfp.pdf(P_H))
        cdf.append(hfp.cdf(P_H))
        tail.append(hfp.tail(P_H))


    for i in range(len(pdf)-1):
        np.testing.assert_allclose(pdf[i], pdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(cdf[i], cdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(tail[i], tail[-1], rtol=_PREC_RTOL[i], atol=0.0)

    np.testing.assert_allclose(tail[-1], 1.0 - cdf[-1], rtol=0.0, atol=5e-16)


def test_across_precisions_N1000_no_anomaly():
    """
    Use different numerical precisions to cross-validate the
    numerics.
    """
    print("test_across_precisions_N1000")
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

    hfps = []
    for prec in ['double','long double']:
        print("prec:",prec)
        try:
            hfps.append(
                HeatFlowAnomalyPosterior(q, x, y, ano, gcp, 5.0e3,
                                        precision=prec,
                                        bli_max_refinements=5)
            )
        except Exception as e:
            from pickle import Pickler
            with open('dump.pickle','wb') as f:
                Pickler(f).dump((q, x, y))
            raise e

    P_H = np.linspace(0, max(hfp.PHmax_global for hfp in hfps), 200)
    P_H = np.linspace(0, max(hfp.PHmax_global for hfp in hfps), 20)

    pdf = []
    cdf = []
    tail = []
    for hfp in hfps:
        pdf.append(hfp.pdf(P_H))
        cdf.append(hfp.cdf(P_H))
        tail.append(hfp.tail(P_H))


    for i in range(len(pdf)-1):
        np.testing.assert_allclose(pdf[i], pdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(cdf[i], cdf[-1], rtol=_PREC_RTOL[i], atol=0.0)
        np.testing.assert_allclose(tail[i], tail[-1], rtol=_PREC_RTOL[i], atol=0.0)

    np.testing.assert_allclose(tail[-1], 1.0 - cdf[-1], rtol=0.0, atol=5e-16)


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



def test_absolute_values_2():
    """
    A second set of absolute test values evaluated with MPMath.
    This set of values stems from an error occuring in a previous
    code version.
    """
    # Problem description (synthetic data and anomaly):
    qc_i = np.array([
        [74.899297645192660866,9.0062284112191297855e-09],
        [65.059810425317650129,6.292979631087474626e-08],
        [73.7333051571991831,1.7312867537817080333e-07],
        [32.101166178978616017,3.3383770134386603678e-08],
        [70.933876424473609745,9.3611019395612739747e-09],
        [27.975606558522464695,1.0604084288016606936e-08],
        [51.91965658284368601,5.989265793684179317e-09],
        [66.893346975152468303,3.3658297443567555587e-09],
        [65.329735427875334608,1.6897876188526839879e-08],
        [34.859875907287054986,8.0360194318523393276e-09],
    ])
    PH_true = 100e6
    qi = qc_i[:,0].copy()
    ci = qc_i[:,1].copy()
    qi_ano = qi + PH_true * ci

    # Reference posterior results using MPMath:
    p,s,n,v = 1.0, 0.0, 0.0, 0.0

    PH = np.array([
        5.2588415072838783e+08, 5.2588423163412720e+08, 5.2588431253986663e+08,
        5.2588439344560599e+08, 5.2588447435134542e+08, 5.2588455525708479e+08,
        5.2588463616282415e+08, 5.2588471706856358e+08, 5.2588479797430295e+08,
        5.2588487888004237e+08, 5.2588495978578174e+08, 5.2588504069152117e+08,
        5.2588512159726053e+08, 5.2588520250299990e+08, 5.2588528340873933e+08,
        5.2588536431447870e+08, 5.2588544522021812e+08, 5.2588552612595749e+08,
        5.2588560703169686e+08, 5.2588568793743628e+08, 5.2588576884317565e+08,
        5.2588584974891508e+08, 5.2588593065465444e+08, 5.2588601156039381e+08,
        5.2588609246613324e+08, 5.2588617337187260e+08, 5.2588625427761203e+08,
        5.2588633518335140e+08, 5.2588641608909076e+08, 5.2588649699483019e+08,
        5.2588657790056956e+08, 5.2588665880630898e+08, 5.2588673971204835e+08,
        5.2588682061778778e+08, 5.2588690152352715e+08, 5.2588698242926651e+08,
        5.2588706333500594e+08, 5.2588714424074531e+08, 5.2588722514648473e+08,
        5.2588730605222410e+08])
    PDF_ref = np.array([
        1.3804808639347031e-12, 1.3754277761088464e-12, 1.3702780052325862e-12,
        1.3650232241800211e-12, 1.3596618735839209e-12, 1.3541853334923251e-12,
        1.3485898274158802e-12 ,1.3428680210209298e-12, 1.3370124853858134e-12,
        1.3310157723051208e-12, 1.3248695188875157e-12, 1.3185642924881516e-12,
        1.3120896695221981e-12, 1.3054380019744987e-12, 1.2985869346683246e-12,
        1.2915262233344576e-12, 1.2842442537857658e-12, 1.2767253038125161e-12,
        1.2689495431452287e-12, 1.2608930334699183e-12, 1.2525277284467536e-12,
        1.2438214736679289e-12, 1.2347380067070772e-12, 1.2252369570749971e-12,
        1.2152738462375073e-12, 1.2048000876388538e-12, 1.1937453514560331e-12,
        1.1820172863788565e-12, 1.1695162660449624e-12, 1.1561108748402565e-12,
        1.1416311010795885e-12, 1.1258510921174624e-12, 1.1084608925980694e-12,
        1.0890173300551590e-12, 1.0668518867323074e-12, 1.0408797808996052e-12,
        1.0091445750486844e-12, 9.6746735343595499e-13, 9.0345495624958327e-13,
        2.9236779560494852e-13])


    # Compare the three posterior PDF backends:
    post0 = CppAnomalyPosterior([qi_ano], [ci], np.ones(1),
                                p, s, n, v, 1.0, 1e-8,
                                pdf_algorithm='barycentric_lagrange')
    post1 = CppAnomalyPosterior([qi_ano], [ci], np.ones(1),
                                p, s, n, v, 1.0, 1e-8,
                                pdf_algorithm='adaptive_simpson')
    post2 = CppAnomalyPosterior([qi_ano], [ci], np.ones(1),
                                p, s, n, v, 1.0, 1e-8,
                                pdf_algorithm='explicit')

    assert np.allclose(post0.pdf(PH), PDF_ref)
    assert np.allclose(post1.pdf(PH), PDF_ref)
    assert np.allclose(post2.pdf(PH), PDF_ref)


def test_high_precision():
    support_float128()
    support_dec50()
    support_dec100()


    # Problem description (synthetic data and anomaly):
    qc_i = np.array([
        [74.899297645192660866,9.0062284112191297855e-09],
        [65.059810425317650129,6.292979631087474626e-08],
        [73.7333051571991831,1.7312867537817080333e-07],
        [32.101166178978616017,3.3383770134386603678e-08],
        [70.933876424473609745,9.3611019395612739747e-09],
        [27.975606558522464695,1.0604084288016606936e-08],
        [51.91965658284368601,5.989265793684179317e-09],
        [66.893346975152468303,3.3658297443567555587e-09],
        [65.329735427875334608,1.6897876188526839879e-08],
        [34.859875907287054986,8.0360194318523393276e-09],
    ])
    PH_true = 100e6
    qi = qc_i[:,0].copy()
    ci = qc_i[:,1].copy()
    qi_ano = qi + PH_true * ci

    # Reference posterior results using MPMath:
    p,s,n,v = 1.0, 0.0, 0.0, 0.0

    PH = np.array([
        5.2588415072838783e+08, 5.2588423163412720e+08, 5.2588431253986663e+08,
        5.2588439344560599e+08, 5.2588447435134542e+08, 5.2588455525708479e+08,
        5.2588463616282415e+08, 5.2588471706856358e+08, 5.2588479797430295e+08,
        5.2588487888004237e+08, 5.2588495978578174e+08, 5.2588504069152117e+08,
        5.2588512159726053e+08, 5.2588520250299990e+08, 5.2588528340873933e+08,
        5.2588536431447870e+08, 5.2588544522021812e+08, 5.2588552612595749e+08,
        5.2588560703169686e+08, 5.2588568793743628e+08, 5.2588576884317565e+08,
        5.2588584974891508e+08, 5.2588593065465444e+08, 5.2588601156039381e+08,
        5.2588609246613324e+08, 5.2588617337187260e+08, 5.2588625427761203e+08,
        5.2588633518335140e+08, 5.2588641608909076e+08, 5.2588649699483019e+08,
        5.2588657790056956e+08, 5.2588665880630898e+08, 5.2588673971204835e+08,
        5.2588682061778778e+08, 5.2588690152352715e+08, 5.2588698242926651e+08,
        5.2588706333500594e+08, 5.2588714424074531e+08, 5.2588722514648473e+08,
        5.2588730605222410e+08])
    PDF_ref = np.array([
        1.3804808639347031e-12, 1.3754277761088464e-12, 1.3702780052325862e-12,
        1.3650232241800211e-12, 1.3596618735839209e-12, 1.3541853334923251e-12,
        1.3485898274158802e-12 ,1.3428680210209298e-12, 1.3370124853858134e-12,
        1.3310157723051208e-12, 1.3248695188875157e-12, 1.3185642924881516e-12,
        1.3120896695221981e-12, 1.3054380019744987e-12, 1.2985869346683246e-12,
        1.2915262233344576e-12, 1.2842442537857658e-12, 1.2767253038125161e-12,
        1.2689495431452287e-12, 1.2608930334699183e-12, 1.2525277284467536e-12,
        1.2438214736679289e-12, 1.2347380067070772e-12, 1.2252369570749971e-12,
        1.2152738462375073e-12, 1.2048000876388538e-12, 1.1937453514560331e-12,
        1.1820172863788565e-12, 1.1695162660449624e-12, 1.1561108748402565e-12,
        1.1416311010795885e-12, 1.1258510921174624e-12, 1.1084608925980694e-12,
        1.0890173300551590e-12, 1.0668518867323074e-12, 1.0408797808996052e-12,
        1.0091445750486844e-12, 9.6746735343595499e-13, 9.0345495624958327e-13,
        2.9236779560494852e-13])


    # Compare the three posterior PDF backends:
    if support_float128():
        post0 = CppAnomalyPosterior([qi_ano], [ci], np.ones(1),
                                    p, s, n, v, 1.0, _PREC_RTOL_FLOAT128,
                                    precision="float128")
        assert np.allclose(post0.pdf(PH), PDF_ref)
    else:
        warn("REHEATFUNQ compiled without support for boost::multiprecision::float128")

    if support_dec50():
        post1 = CppAnomalyPosterior([qi_ano], [ci], np.ones(1),
                                    p, s, n, v, 1.0, _PREC_RTOL_DEC50,
                                    precision="dec50")
        assert np.allclose(post1.pdf(PH), PDF_ref)
    else:
        warn("REHEATFUNQ compiled without support for boost::multiprecision::dec50")

    if support_dec100():
        post2 = CppAnomalyPosterior([qi_ano], [ci], np.ones(1),
                                    p, s, n, v, 1.0, _PREC_RTOL_DEC100,
                                    precision="dec100")
        assert np.allclose(post2.pdf(PH), PDF_ref)
    else:
        warn("REHEATFUNQ compiled without support for boost::multiprecision::dec100")
