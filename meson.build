project('REHEATFUNQ','cpp','cython', version : '1.4.0',
        default_options : ['optimization=3'])
add_project_arguments(['-std=c++20','-msse3'], language : 'cpp')

# Use natively tuned code by default - turn off by use of the 'portable'
# option:
if not get_option('portable')
    add_project_arguments(['-march=native','-mtune=native'], language : 'cpp')
endif

include = include_directories('include','external/pdtoolbox/include')

eigen_dep = dependency('eigen3', required : false)
if not eigen_dep.found()
    # This is a patch for the Docker build.
    # Eigen does not really need any installed library as
    # long as all the headers can be found.
    # If the dependency lookup failed, we proceed in good
    # faith that the headers are installed in the correct
    # position.
    compiler = meson.get_compiler('cpp')
    if not compiler.check_header('eigen3/Eigen/Dense',
                                 include_directories : include)
        error('Could not access the Eigen headers. Make sure that from your ' +
              'include path, "eigen3/Eigen/Dense" can be accessed.')
    endif
endif
boost_dep = dependency('boost')
openmp_dep = dependency('openmp')
geographiclib_dep = dependency('GeographicLib', required : false)
if not geographiclib_dep.found()
    # This is a potential patch for Ubuntu systems where
    # the FindGeographicLib.cmake is located wrongly:
    geographiclib_dep = dependency(
        'GeographicLib',
        method : 'cmake',
        cmake_module_path : '/usr/share/cmake/geographiclib/'
    )
endif


#
# Find the NumPy headers here:
#
python = import('python').find_installation()
dep_py = python.dependency()

incpath_np = run_command(
  python,
  ['numpy-include.py'],
  check : false
)
if incpath_np.returncode() != 0
    message(incpath_np.stdout().strip())
    message(incpath_np.stderr().strip())
    error('Could not determine NumPy include path.')
else
    incpath_np = incpath_np.stdout().strip()
endif

incdir_np = include_directories([incpath_np, incpath_np / '..' / '..' / '..'])

add_project_arguments(['-I' + incpath_np + '/../../../'], language : 'cython')

################################################################################
#                                                                              #
#                             BUILD TARGETS                                    #
#                                                                              #
################################################################################


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
# R-disks:
#
sources_rdisks = ['src/coverings/dminpermutations.cpp']
dependencies_rdisks = [boost_dep]
librdisks = static_library(
    'rdisks',
    sources_rdisks,
    dependencies : dependencies_rdisks,
    include_directories : include
)


#
# Anomaly:
#
sources_anomaly = ['src/anomaly.cpp', 'src/anomaly/ls1980.cpp',
                   'src/anomaly/nearestneighbor.cpp']
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

hfp_cpp_args = ['-DBOOST_ENABLE_ASSERT_HANDLER']
hfp_link_args = []
if get_option('anomaly_posterior_float128')
    hfp_cpp_args += ['--std=gnu++20','-DREHEATFUNQ_ANOMALY_POSTERIOR_TYPE_QUAD']
    hfp_link_args += ['-lquadmath']
endif
if get_option('anomaly_posterior_dec50')
    hfp_cpp_args += ['-DREHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_50']
endif
if get_option('anomaly_posterior_dec100')
    hfp_cpp_args += ['-DREHEATFUNQ_ANOMALY_POSTERIOR_TYPE_BOOST_DEC_100']
endif


libcommon = static_library(
    'common', ['src/numerics/boostassert.cpp'],
    dependencies : dependencies_hfp,
    cpp_args : hfp_cpp_args,
    link_args : hfp_link_args,
    include_directories : include
)

libhfp = static_library(
    'hfp', sources_hfp,
    dependencies : dependencies_hfp,
    cpp_args : hfp_cpp_args,
    link_args : hfp_link_args,
    link_with : libcommon,
    include_directories : include
)

libhfp2 = static_library(
    'hfp2', ['src/anomaly/variableprecisionposterior.cpp',
             'src/anomaly/posterior.cpp',
             'src/anomaly/posterior/integrand.cpp'],
    dependencies : dependencies_hfp,
    cpp_args : hfp_cpp_args,
    link_args : hfp_link_args,
    link_with : libcommon,
    include_directories : include
)


#
# Resilience:
#
include_resilience =  [include, 'external/zeal2022hfpc/include']
sources_resilience = ['external/zeal2022hfpc/src/synthetic_covering.cpp',
                      'src/resilience/resilience.cpp']
dependencies_resilience = [openmp_dep, boost_dep]

libresilience = static_library(
    'resilience', sources_resilience,
    dependencies : dependencies_resilience,
    include_directories : include_resilience
)


#
# Testing:
#
sources_testing = ['src/testing/barylagrint.cpp']
libtesting = static_library(
    'testing', sources_testing,
    dependencies : [openmp_dep, boost_dep],
    include_directories : include
)


#
# Python function wrapper:
#
sources_pyfunwrap = ['src/utility/pyfunwrap.cpp']
libpyfunwrap = static_library(
    'pyfunwrap', sources_pyfunwrap,
    dependencies : [dep_py],
    include_directories : include
)


message('NumPy include directory: "' + incpath_np + '".')

# Numpy link dependencies:
numpy_random = meson.get_compiler('cpp').find_library(
                            'npyrandom',
                            dirs: [incpath_np / '..' / '..' / 'random' / 'lib',
                                   incpath_np / '..' / '..' / '..']
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
                        link_with : libhfp,
                        link_args : hfp_link_args)

python.extension_module('postbackend',
                        'reheatfunq/anomaly/postbackend.pyx',
                        dependencies : [dep_py, dependencies_hfp],
                        include_directories : include,
                        override_options : ['cython_language=cpp'],
                        link_with : [libhfp2, libcommon, libhfp],
                        cpp_args : hfp_cpp_args,
                        link_args : hfp_link_args)

python.extension_module('rdisks',
                        'reheatfunq/coverings/rdisks.pyx',
                        dependencies : [dep_py, numpy_random, numpy_math],
                        include_directories : [include, incdir_np],
                        override_options : ['cython_language=cpp'],
                        link_with : librdisks
                        )

python.extension_module('poisampling',
                        'reheatfunq/coverings/poisampling.pyx',
                        dependencies : [dep_py, numpy_random, numpy_math],
                        include_directories : [include, incdir_np],
                        override_options : ['cython_language=cpp']
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
                        link_with : [libhfp2, libresilience, libhfp],
                        cpp_args : hfp_cpp_args,
                        link_args : hfp_link_args
                        )

python.extension_module('distancedistribution',
                        'reheatfunq/data/distancedistribution.pyx',
                        dependencies : [dep_py, geographiclib_dep],
                        override_options : ['cython_language=cpp'],
                        )


python.extension_module('barylagrint',
                        'reheatfunq/_testing/barylagrint.pyx',
                        dependencies : [dep_py],
                        override_options : ['cython_language=cpp'],
                        include_directories : [include],
                        link_with : [libtesting, libpyfunwrap]
                        )