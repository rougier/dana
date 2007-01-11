#!/usr/bin/env python


# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


import glob
from distutils.core import setup, Extension
import distutils.sysconfig
import numpy

core_srcs = glob.glob ("dana/core/*.cc")
core_ext = Extension (
    'dana.core._core',
    sources = core_srcs,
    include_dirs=[numpy.get_include()],
    libraries = ['boost_python', 'boost_thread']
)

random_srcs = glob.glob ("dana/random/*.cc")
random_ext = Extension (
    'dana.random._random',
    sources = random_srcs,
    include_dirs=[numpy.get_include()],
    libraries = ['boost_python'],
)

profile_srcs = glob.glob ("dana/projection/profile/*.cc")
profile_ext = Extension (
    'dana.projection.profile._profile',
    sources = profile_srcs,
    include_dirs=[numpy.get_include()],
    libraries = ['boost_python']
)

density_srcs = glob.glob ("dana/projection/density/*.cc")
density_ext = Extension (
    'dana.projection.density._density',
    sources = density_srcs,
    include_dirs=[numpy.get_include()],
    libraries = ['boost_python']
)

distance_srcs = glob.glob ("dana/projection/distance/*.cc")
distance_ext = Extension (
    'dana.projection.distance._distance',
    sources = distance_srcs,
    include_dirs=[numpy.get_include()],
    libraries = ['boost_python']
)

shape_srcs = glob.glob ("dana/projection/shape/*.cc")
shape_ext = Extension (
    'dana.projection.shape._shape',
    sources = shape_srcs,
    include_dirs=[numpy.get_include()],
    libraries = ['boost_python']
)

projection_srcs = glob.glob ("dana/projection/*.cc")
projection_ext = Extension (
    'dana.projection._projection',
    sources = projection_srcs,
    include_dirs=[numpy.get_include()],
    libraries = ['boost_python', 'boost_thread']
)

setup (name='dana',
       version = '1.0',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description =
            "Distributed Asynchronous Numerical Adaptive computing library",
       packages = ['dana',
                   'dana.core',
                   'dana.random',
                   'dana.view',
                   'dana.projection',
                   'dana.projection.density',
                   'dana.projection.profile',
                   'dana.projection.distance',
                   'dana.projection.shape'],
       ext_modules = [
            core_ext,
            random_ext,
            projection_ext, profile_ext, density_ext, distance_ext, shape_ext
       ],
      
       data_files= [
        ("include/dana/core", glob.glob("dana/core/*.h")),
       ]
      )
