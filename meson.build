project('REHEATFUNQ','cpp','cython',
        default_options : ['optimization=3'])
add_project_arguments(['-march=native','-mtune=native',
                       '-std=c++20'], language : 'cpp')

include = include_directories('include','external/pdtoolbox/include')

eigen_dep = dependency('eigen3')
boost_dep = dependency('boost')
openmp_dep = dependency('openmp')
geographiclib_dep = dependency('GeographicLib')


#
# Conjugate prior:
#
sources = ['external/pdtoolbox/src/ll_gamma_conjugate_prior.cpp',
           'external/pdtoolbox/src/optimize.cpp',
           'external/pdtoolbox/src/gamma.cpp',
           'external/pdtoolbox/src/gamma_conjugate_prior.cpp']

dependencies = [openmp_dep, eigen_dep]

libgcp = static_library(
    'gcp', sources,
    dependencies : dependencies,
    include_directories : include
)


#
# Anomaly:
#
sources_anomaly = ['src/anomaly.cpp', 'src/anomaly/ls1980.cpp']
dependencies_anomaly = [openmp_dep, boost_dep]

libanomaly = static_library(
    'anomaly', sources_anomaly,
    dependencies : dependencies_anomaly,
    include_directories : include
)


#
# Heat flow posterior:
#
sources_hfp = ['external/pdtoolbox/src/ziebarth2022a.cpp',
               'external/pdtoolbox/src/quantileinverter.cpp']
dependencies_hfp = [openmp_dep, boost_dep]

libhfp = static_library(
    'hfp', sources_hfp,
    dependencies : dependencies_hfp,
    include_directories : include
)


#
# Resilience:
#
include_resilience =  [include, 'external/zeal2022hfpc/include']
sources_resilience = ['external/zeal2022hfpc/src/resilience.cpp',
                      'external/zeal2022hfpc/src/synthetic_covering.cpp']
dependencies_resilience = [openmp_dep, boost_dep]

libresilience = static_library(
    'resilience', sources_resilience,
    dependencies : dependencies_resilience,
    include_directories : include_resilience
)



python = import('python').find_installation()
dep_py = python.dependency()

incpath_np = run_command(
  python,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()
incdir_np = include_directories(incpath_np)


# Numpy link dependencies:
numpy_random = meson.get_compiler('cpp').find_library(
                            'npyrandom',
                            dirs: incpath_np / '..' / '..' / 'random' / 'lib'
)
numpy_math = meson.get_compiler('cpp').find_library(
                            'npymath',
                            dirs: incpath_np / '..' / 'lib'
)



python.extension_module('backend',
                        'reheatfunq/regional/backend.pyx',
                        dependencies : [dep_py],
                        include_directories : [include, incdir_np],
                        override_options : ['cython_language=cpp'],
                        link_with : libgcp)

python.extension_module('bayes',
                        'reheatfunq/anomaly/bayes.pyx',
                        dependencies : [dep_py],
                        include_directories : include,
                        override_options : ['cython_language=cpp'],
                        link_with : libhfp)

python.extension_module('rdisks',
                        'reheatfunq/coverings/rdisks.pyx',
                        dependencies : [dep_py, numpy_random, numpy_math],
                        include_directories : [include, incdir_np],
                        override_options : ['cython_language=cpp'],
                        #link_with : libgcp
                        )

python.extension_module('anomaly',
                        'reheatfunq/anomaly/anomaly.pyx',
                        dependencies : [dep_py],
                        include_directories : [include],
                        override_options : ['cython_language=cpp'],
                        link_with : libanomaly
                        )

python.extension_module('zeal2022hfresil',
                        'reheatfunq/resilience/zeal2022hfresil.pyx',
                        dependencies : [dep_py],
                        include_directories : include_resilience,
                        override_options : ['cython_language=cpp'],
                        link_with : [libhfp, libresilience, ]
                        )

python.extension_module('distancedistribution',
                        'reheatfunq/data/distancedistribution.pyx',
                        dependencies : [dep_py, geographiclib_dep],
                        override_options : ['cython_language=cpp'],
                        )