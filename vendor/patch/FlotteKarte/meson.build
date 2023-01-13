project('flottekarte', 'cpp')
add_project_arguments('-O3',  language : 'cpp')

# Use natively tuned code by default - turn off by use of the 'portable'
# option:
if not get_option('portable')
    add_project_arguments(['-march=native','-mtune=native'], language : 'cpp')
endif

# Projwrapper subprocject:
libprojwrap_proj = subproject('libprojwrap')
libprojwrap_dep = libprojwrap_proj.get_variable('libprojwrap_dep')
include = [include_directories('/usr/include'), include_directories('include')]
dependencies = [dependency('openmp'),libprojwrap_dep]
sources = ['src/pyapi.cpp', 'src/invert.cpp',
           'src/types.cpp', 'src/linalg.cpp', 'src/griddedinverter.cpp',
           'src/gradient.cpp', 'src/tickfinder.cpp', 'src/grid.cpp',
           'src/paths.cpp', 'src/boundary.cpp', 'src/augmentedproj.cpp']
library('flottekarte',
        sources : sources,
        include_directories : include,
        dependencies : dependencies,
        link_args : '-Wl,-lproj',
        override_options : ['cpp_std=c++20'])
