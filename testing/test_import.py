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

def test_api_import():
    from reheatfunq import AnomalyLS1980, HeatFlowAnomalyPosterior, \
                           HeatFlowPredictive, GammaConjugatePrior
    from reheatfunq.data import read_nghf
    from reheatfunq.coverings import random_global_R_disk_coverings,\
                           bootstrap_data_selection, conforming_data_selection
    from reheatfunq.regional import gamma_mle


def test_cython_import():
    """
    This function tests whether any of the compiled Cython modules
    throw an import error when imported from. This usually means that
    there is an undefined symbol and meson.build or the source code
    has to be updated.
    """
    from reheatfunq.regional.backend import gamma_conjugate_prior_mle
    from reheatfunq.anomaly.bayes import marginal_posterior_tail
    from reheatfunq.anomaly.postbackend import CppAnomalyPosterior
    from reheatfunq.coverings.rdisks import all_restricted_samples
    from reheatfunq.coverings.poisampling import generate_point_of_interest_sampling
    from reheatfunq.anomaly.anomaly import Anomaly
    from reheatfunq.resilience.zeal2022hfresil \
        import generate_synthetic_heat_flow_coverings_mix2
    from reheatfunq.data.distancedistribution import distance_distribution
    from reheatfunq._testing.barylagrint import BarycentricLagrangeInterpolator