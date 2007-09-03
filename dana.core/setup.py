#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
# 
# $Id: setup.py 275 2007-08-14 15:01:41Z rougier $
#-------------------------------------------------------------------------------

# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import sys, glob
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import distutils.sysconfig
import commands


# ___________________________________________________________check configuration
print "============================================================"
print "Checking configuration"
print

# _________________________________________________________________________numpy
sys.stdout.write ("Checking for numpy package... ")
try:
    import numpy
except:
    print "failed"
    print " -> Consider installing numpy (http://numpy.scipy.org/)"
    sys.exit(1)
else:
    print "ok"

# _________________________________________________________________________boost
print "Checking for boost libraries... unknown status"
print " -> Make sure boost-python library has been installed"
print " -> Make sure boost-thread library has been installed"
boost_python_lib = 'boost_python-mt'
boost_thread_lib = 'boost_thread-mt'

# _______________________________________________________________________libxml2
libxml2_include_path = ""
sys.stdout.write ("Checking for libxml2...")
(status, text) = commands.getstatusoutput("pkg-config xml2po --exists")
if status == 0:
    print "ok"
    libxml2_include_path = commands.getoutput("pkg-config xml2po --cflags")[2:]
else:
    print "failed"
    print " -> Consider installing libxml2 (http://xmlsoft.org/)"
    sys.exit(1)

print "============================================================"





# __________________________________________________________________________core
core_lib_srcs = glob.glob ("dana/core/*.cc")
core_lib_srcs = filter(lambda x: not 'core.cc' in x, core_lib_srcs)
core_lib = Extension (
    'dana.libdana_core',
    sources = core_lib_srcs,
    include_dirs=[numpy.get_include(), libxml2_include_path],
    libraries = [boost_thread_lib, 'xml2']
)

core_ext_srcs = glob.glob ("dana/core/core.cc")
core_ext = Extension (
    'dana.core._core',
    sources = core_ext_srcs,
    include_dirs=[numpy.get_include(), libxml2_include_path],
    libraries = [boost_python_lib, boost_thread_lib, 'xml2', 'dana_core']
)


# _______________________________________________________________________profile
profile_lib_srcs = glob.glob ("dana/projection/profile/*.cc")
profile_lib_srcs = filter(lambda x: not 'profile_export.cc' in x, profile_lib_srcs)
profile_lib = Extension (
    'dana.libdana_projection_profile',
    sources      = profile_lib_srcs,
    include_dirs =[numpy.get_include(), libxml2_include_path],
    libraries    = ['xml2', 'dana_core']
)
profile_ext_srcs = glob.glob ("dana/projection/profile/profile_export.cc")
profile_ext = Extension (
    'dana.projection.profile._profile',
    sources = profile_ext_srcs,
    include_dirs=[numpy.get_include(), libxml2_include_path],
    libraries = [boost_python_lib, 'xml2', 'dana_projection_profile', 'dana_core']
)

# _______________________________________________________________________distance
distance_lib_srcs = glob.glob ("dana/projection/distance/*.cc")
distance_lib_srcs = filter(lambda x: not 'distance_export.cc' in x, distance_lib_srcs)
distance_lib = Extension (
    'dana.libdana_projection_distance',
    sources      = distance_lib_srcs,
    include_dirs =[numpy.get_include(), libxml2_include_path],
    libraries    = ['xml2', 'dana_core']
)
distance_ext_srcs = glob.glob ("dana/projection/distance/distance_export.cc")
distance_ext = Extension (
    'dana.projection.distance._distance',
    sources = distance_ext_srcs,
    include_dirs=[numpy.get_include(), libxml2_include_path],
    libraries = [boost_python_lib, 'xml2', 'dana_projection_distance', 'dana_core']
)

# _______________________________________________________________________shape
shape_lib_srcs = glob.glob ("dana/projection/shape/*.cc")
shape_lib_srcs = filter(lambda x: not 'shape_export.cc' in x, shape_lib_srcs)
shape_lib = Extension (
    'dana.libdana_projection_shape',
    sources      = shape_lib_srcs,
    include_dirs =[numpy.get_include(), libxml2_include_path],
    libraries    = ['xml2', 'dana_core']
)
shape_ext_srcs = glob.glob ("dana/projection/shape/shape_export.cc")
shape_ext = Extension (
    'dana.projection.shape._shape',
    sources = shape_ext_srcs,
    include_dirs=[numpy.get_include(), libxml2_include_path],
    libraries = [boost_python_lib, 'xml2', 'dana_projection_shape', 'dana_core']
)

# _______________________________________________________________________density
density_lib_srcs = glob.glob ("dana/projection/density/*.cc")
density_lib_srcs = filter(lambda x: not 'density_export.cc' in x, density_lib_srcs)
density_lib = Extension (
    'dana.libdana_projection_density',
    sources      = density_lib_srcs,
    include_dirs =[numpy.get_include(), libxml2_include_path],
    libraries    = ['xml2', 'dana_core']
)
density_ext_srcs = glob.glob ("dana/projection/density/density_export.cc")
density_ext = Extension (
    'dana.projection.density._density',
    sources = density_ext_srcs,
    include_dirs=[numpy.get_include(), libxml2_include_path],
    libraries = [boost_python_lib, 'xml2', 'dana_projection_density', 'dana_core']
)



# ____________________________________________________________________projection
projection_ext_srcs = glob.glob ("dana/projection/*.cc")
projection_ext = Extension (
    'dana.projection._projection',
    sources = projection_ext_srcs,
    include_dirs=[numpy.get_include(), libxml2_include_path],
    libraries = [boost_python_lib,
                 'dana_core',
                 'dana_projection_distance',
                 'dana_projection_density',
                 'dana_projection_shape',
                 'dana_projection_profile']
                 
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
        extra_dir = os.path.join(extra_dir, 'dana')
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
            core_lib,     core_ext,
            profile_lib,  profile_ext,
            density_lib,  density_ext,
            distance_lib, distance_ext,
            shape_lib,    shape_ext,
            projection_ext
       ],
       cmdclass = {
		"build_ext": my_build_ext
	   },
       data_files= [
        ("bin", ("bin/gpython",)),
        ("include/dana/core", glob.glob("dana/core/*.h"))
       ]
      )

print
print "======================================================================="
print
print " You have to modify your LD_LIBRARY_PATH environment variable by adding"
print "  $prefix + 'python%s/site-packages/dana'" % sys.version[0:3]
print
print "======================================================================="
print
