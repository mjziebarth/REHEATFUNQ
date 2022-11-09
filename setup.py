# Setup script.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Malte J. Ziebarth


from setuptools import setup, Extension
from mebuex import MesonExtension, build_ext

regional_backend    = MesonExtension('reheatfunq.regional.backend')
anomaly_bayes       = MesonExtension('reheatfunq.anomaly.bayes')
anomaly_anomaly     = MesonExtension('reheatfunq.anomaly.anomaly')
coverings_rdisks    = MesonExtension('reheatfunq.coverings.rdisks')
resilience_zeal2022 = MesonExtension('reheatfunq.resilience.zeal2022hfresil')


setup(name='REHEATFUNQ',
      version='1.0.0',
      author='Malte J. Ziebarth',
      description='',
      packages = ['reheatfunq','reheatfunq.regional','reheatfunq.anomaly',
                  'reheatfunq.data', 'reheatfunq.coverings',
                  'reheatfunq.resilience'],
      ext_modules=[regional_backend, anomaly_bayes, coverings_rdisks,
                   anomaly_anomaly, resilience_zeal2022],
      cmdclass={'build_ext' : build_ext}
)

