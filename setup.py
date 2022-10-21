# Setup script.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Malte J. Ziebarth


from setuptools import setup, Extension
from mebuex import MesonExtension, build_ext

regional_backend = MesonExtension('reheatfunq.regional.backend')

anomaly_bayes = MesonExtension('reheatfunq.anomaly.bayes')


setup(name='REHEATFUNQ',
      version='1.0.0',
      author='Malte J. Ziebarth',
      description='',
      packages = ['reheatfunq','reheatfunq.regional','reheatfunq.anomaly'],
      ext_modules=[regional_backend, anomaly_bayes],
      cmdclass={'build_ext' : build_ext}
)

