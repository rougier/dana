#!/usr/bin/env python

#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
#------------------------------------------------------------------------------


# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


import sys,glob
from distutils.core import setup
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.command.install_data import install_data
import distutils.sysconfig
import numpy

# Get core and cnft shared object filename
from dana.core._core import __file__ as core
from dana.cnft._cnft import __file__ as cnft

# Modify include_dirs
include_dirs = []
include_dirs.append (numpy.get_include())
include_tmp = core[:core.find('python')]
include_dirs.append(os.path.normpath(os.path.join(include_tmp,'../include/dana/')))



libsigmapi_srcs = glob.glob("dana/sigmapi/*.cc")
libsigmapi_srcs = filter(lambda x: not 'sigmapi.cc' in x, libsigmapi_srcs)
libsigmapi_lib = Extension (
    'dana.libsigmapi',
    sources = libsigmapi_srcs,
    include_dirs = include_dirs,
    libraries = ['boost_python'],
    extra_objects= [core,cnft]
)
sigmapi_srcs = glob.glob ("dana/sigmapi/sigmapi.cc")
sigmapi_ext = Extension (
    'dana.sigmapi._sigmapi',
    sources = sigmapi_srcs,
    include_dirs = include_dirs,
    libraries = ['boost_python','sigmapi'],
    extra_objects = [core,cnft]
)

projection_srcs = glob.glob ("dana/sigmapi/projection/*.cc")
projection_ext = Extension (
    'dana.sigmapi.projection._projection',
    sources = projection_srcs,
    libraries = ['boost_python','sigmapi'],
    include_dirs =  include_dirs,
    extra_objects=[core,cnft]
)

combination_srcs = glob.glob ("dana/sigmapi/projection/combination/*.cc")
combination_ext = Extension (
    'dana.sigmapi.projection.combination._combination',
    sources = combination_srcs,
    libraries = ['boost_python','sigmapi'],
    include_dirs =  include_dirs,
    extra_objects=[core,cnft]
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
    def build_extension(self, ext):
        force_optimisation(self.compiler)
        extra_dir = self.build_lib
        extra_dir = os.path.join(extra_dir, 'dana')
        ext.library_dirs.append(extra_dir)
        build_ext.build_extension(self, ext)

class my_install_data(install_data):
    def finalize_options (self):
        self.set_undefined_options ('install', ('install_lib', 'install_dir'))
        install_data.finalize_options(self)

#______________________________________________________________________________
s = setup (name='dana.sigmapi',
       version = '1.0',
       author = 'Jeremy Fix',
       author_email = 'Jeremy.Fix@loria.fr',
       url = 'http://www.loria.fr/~fix',
       description = "DANA: Sigma Pi Neurons",
       ext_modules = [libsigmapi_lib,
                      sigmapi_ext,
                      projection_ext,
                      combination_ext],       
       packages = ['dana.sigmapi',
                   'dana.sigmapi.projection',
                   'dana.sigmapi.projection.combination'],
       cmdclass = { 'install_data' : my_install_data,
                    'build_ext': my_build_ext},
       data_files = [("include/dana/sigmapi",glob.glob('dana/sigmapi/*.h')),
                     ("include/dana/sigmapi/projection",glob.glob("dana/sigmapi/projection/*.h")),
                     ("include/dana/sigmapi/projection/combination",glob.glob("dana/sigmapi/projection/combination/*.h"))]
      )

print
print "======================================================================="
print
print " You have to modify your LD_LIBRARY_PATH environment variable by adding"
print "  $prefix + 'python%s/site-packages/dana'" % sys.version[0:3]
print
print "======================================================================="
print

