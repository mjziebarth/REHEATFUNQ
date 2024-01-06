# This script tests all Jupyter notebook imports.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2021 Deutsches GeoForschungsZentrum GFZ,
#               2022-2024 Malte J. Ziebarth
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

#
# Numpy and Scipy:
#
import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse as sp
from scipy.spatial import KDTree
from scipy.optimize import root_scalar
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import quad, dblquad
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator, UnivariateSpline, NearestNDInterpolator, interp1d

#
# Python standard libraries:
#
import os
import json
import requests
from pathlib import Path
from pickle import Pickler, Unpickler
from zipfile import ZipFile
from itertools import product
from math import exp, log2, ceil, acos, cos, sin, degrees, pi
from multiprocessing import cpu_count

#
# REHEATFUNQ:
#
from reheatfunq.data import read_nghf, distance_distribution
from reheatfunq import GammaConjugatePrior, AnomalyLS1980, HeatFlowAnomalyPosterior, HeatFlowPredictive
from reheatfunq.coverings.poisampling import generate_point_of_interest_sampling
from reheatfunq.anomaly.bayes import *
from reheatfunq.anomaly.postbackend import *
from reheatfunq.anomaly import AnomalyLS1980, HeatFlowAnomalyPosterior
from reheatfunq.regional import default_prior
from reheatfunq.coverings import random_global_R_disk_coverings
from reheatfunq.resilience import test_performance_cython, \
                                  test_performance_mixture_cython
from reheatfunq.coverings import random_global_R_disk_coverings
from reheatfunq.coverings import random_global_R_disk_coverings
from reheatfunq.resilience import test_performance_cython
from reheatfunq.regional import default_prior
from reheatfunq.regional.backend import gamma_mle
from reheatfunq.coverings import random_global_R_disk_coverings
from reheatfunq.resilience import generate_synthetic_heat_flow_coverings_mix3, \
                                  generate_normal_mixture_errors_3

#
# PDToolbox:
#
from pdtoolbox.mle import *
from pdtoolbox.likelihood import *
from pdtoolbox.distributions import *
from pdtoolbox.gof.statistics import _anderson_darling
from pdtoolbox.distributions import gamma_cdf
from pdtoolbox import normal_mvue, normal_logL, normal_cdf, \
                      frechet_mle, frechet_logL, frechet_cdf, \
                      gamma_mle, gamma_logL, gamma_cdf, \
                      nakagami_mle, nakagami_logL, nakagami_cdf, \
                      log_logistic_mle, log_logistic_logL, log_logistic_cdf, \
                      shifted_gompertz_mle, shifted_gompertz_logL, shifted_gompertz_cdf, \
                      weibull_mle, weibull_logL, weibull_cdf, \
                      log_normal_mle, log_normal_logL, log_normal_cdf, \
                      inverse_gamma_mle, inverse_gamma_logL, inverse_gamma_cdf, \
                      gamma_pdf, normal_pdf
from pdtoolbox.cython.gamma_accel import gamma_ks_ad_batch
from pdtoolbox.gof import LillieforsTable, AndersonDarlingTable
from pdtoolbox.gof.statistics import _anderson_darling
from pdtoolbox import gamma_pdf, normal_pdf
from pdtoolbox import ExtremeValue2Distribution, FrechetDistribution, \
                      GammaDistribution, InverseGammaDistribution, \
                      LogLogisticDistribution, LogNormalDistribution, \
                      NakagamiDistribution, NormalDistribution, \
                      ShiftedGompertzDistribution, WeibullDistribution

#
# Matplotlib:
#
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection, QuadMesh, PolyCollection
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, Wedge, FancyBboxPatch, Polygon
from matplotlib import rcParams
from matplotlib.ticker import FixedLocator, LogFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm, LogNorm, SymLogNorm, ListedColormap, rgb_to_hsv, hsv_to_rgb

import cmasher
from cmasher import get_sub_cmap
from cmocean.tools import crop
from cmcrameri.cm import *

#
# Geometry and Geostatistics:
#
import shapely
import geopandas as gpd
from pyproj import Proj, Geod
from flottekarte import Map, GeoJSON
from shapely.geometry import Polygon as SPoly, MultiLineString as SMLS, LineString as SL
from sklearn.neighbors import KernelDensity

from joblib import Parallel, delayed, Memory, hash as jhash
import shgofast
from loaducerf3 import UCERF3Model, UCERF3Files, Polygon, PolygonSelector
