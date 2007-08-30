#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: setup.py 275 2007-08-14 15:01:41Z rougier $
#------------------------------------------------------------------------------

# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


import glob
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import distutils.sysconfig
import numpy

boost_python_lib = 'boost_python-gcc41-mt'
boost_thread_lib = 'boost_thread-gcc41-mt'


core_srcs = glob.glob ("dana/core/*.cc")
core_ext = Extension (
    'dana.core._core',
    sources = core_srcs,
    include_dirs=[numpy.get_include(), '/usr/include/libxml2'],
    libraries = [boost_python_lib, boost_thread_lib, 'xml2']
)

profile_srcs = glob.glob ("dana/projection/profile/*.cc")
profile_ext = Extension (
    'dana.projection.profile._profile',
    sources = profile_srcs,
    include_dirs=[numpy.get_include(), '/usr/include/libxml2'],
    libraries = [boost_python_lib]
)

density_srcs = glob.glob ("dana/projection/density/*.cc")
density_ext = Extension (
    'dana.projection.density._density',
    sources = density_srcs,
    include_dirs=[numpy.get_include(), '/usr/include/libxml2'],
    libraries = [boost_python_lib]
)

distance_srcs = glob.glob ("dana/projection/distance/*.cc")
distance_ext = Extension (
    'dana.projection.distance._distance',
    sources = distance_srcs,
    include_dirs=[numpy.get_include(), '/usr/include/libxml2'],
    libraries = [boost_python_lib]
)

shape_srcs = glob.glob ("dana/projection/shape/*.cc")
shape_ext = Extension (
    'dana.projection.shape._shape',
    sources = shape_srcs,
    include_dirs=[numpy.get_include(), '/usr/include/libxml2'],
    libraries = [boost_python_lib]
)

projection_srcs = glob.glob ("dana/projection/*.cc")
projection_ext = Extension (
    'dana.projection._projection',
    sources = projection_srcs,
    include_dirs=[numpy.get_include(), '/usr/include/libxml2'],
    libraries = [boost_python_lib, boost_thread_lib]
)

#______________________________________________________________________________
def force_optimisation(compiler):
    for i in range(4):      
        try: compiler.compiler_so.remove("-O%s" % i)
        except:	pass
    try:
        compiler.compiler_so.remove('-Wstrict-prototypes')
        compiler.compiler_so.remove('-g')
        compiler.compiler_so.remove('-DNDEBUG')
    except:
        pass
    compiler.compiler_so.append("-O3")

class my_build_ext(build_ext):
    def initialize_options(self):
        self.package = None
        build_ext.initialize_options(self)
		
    def finalize_options(self):
        if self.package is None:
            self.package = self.distribution.ext_package
        build_ext.finalize_options(self)
		
    def build_extension(self, ext):
        force_optimisation(self.compiler)
        extra_dir = self.build_lib
        if self.package:
            extra_dir = os.path.join(extra_dir, self.package)
        ext.library_dirs.append(extra_dir)
        build_ext.build_extension(self, ext)

#______________________________________________________________________________
setup (name='dana.core',
       version = '1.0',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description =
            "Distributed Asynchronous Numerical Adaptive computing library",
       packages = ['dana',
                   'dana.core',
                   'dana.tests',                   
                   'dana.projection',
                   'dana.projection.density',
                   'dana.projection.profile',
                   'dana.projection.distance',
                   'dana.projection.shape'
                  ],
       ext_modules = [
            core_ext,
            projection_ext, profile_ext, density_ext, distance_ext, shape_ext
       ],
       cmdclass = {
		"build_ext": my_build_ext
	   },
       data_files= [
        ("bin", ("bin/gpython",)),
        ("include/dana/core", glob.glob("dana/core/*.h"))
       ]
      )
