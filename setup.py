# Setup script.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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


from setuptools import setup, Extension
from mebuex import MesonExtension, build_ext

regional_backend      = MesonExtension('reheatfunq.regional.backend')
anomaly_bayes         = MesonExtension('reheatfunq.anomaly.bayes')
anomaly_postbackend   = MesonExtension('reheatfunq.anomaly.postbackend')
anomaly_anomaly       = MesonExtension('reheatfunq.anomaly.anomaly')
coverings_rdisks      = MesonExtension('reheatfunq.coverings.rdisks')
coverings_poisampling = MesonExtension('reheatfunq.coverings.poisampling')
resilience_zeal2022   = MesonExtension('reheatfunq.resilience.zeal2022hfresil')
data_distancedist     = MesonExtension('reheatfunq.data.distancedistribution')
testing_barylagrint   = MesonExtension('reheatfunq._testing.barylagrint')


setup(ext_modules=[regional_backend, anomaly_bayes, coverings_rdisks,
                   anomaly_anomaly, resilience_zeal2022, data_distancedist,
                   anomaly_postbackend, coverings_poisampling,
                   testing_barylagrint],
      cmdclass={'build_ext' : build_ext}
)
