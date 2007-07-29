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

include_dirs = []
include_dirs.append (numpy.get_include())


libcore_srcs = glob.glob ("glpython/core/*.cc")
libcore_srcs = filter(lambda x: not 'core.cc' in x, libcore_srcs)
libcore_lib = Extension (
    'glpython.libcore',
    sources = libcore_srcs,
    libraries = ['GL', 'GLU', 'ftgl', 'freetype', 'fontconfig']
)
core_srcs = glob.glob ("glpython/core/core.cc")
core_ext = Extension (
    'glpython.core._core',
    sources = core_srcs,
    libraries = ['boost_python', 'GL', 'GLU', 'core']
)

libobjects_srcs = glob.glob ("glpython/objects/*.cc")
libobjects_srcs = filter(lambda x: not 'objects.cc' in x, libobjects_srcs)
libobjects_lib = Extension (
    'glpython.libobjects',
    sources = libobjects_srcs,
    include_dirs = include_dirs,
    libraries = ['GL', 'GLU', 'core', '3ds']
)
objects_srcs = glob.glob ("glpython/objects/*.cc")
objects_ext = Extension (
    'glpython.objects._objects',
    sources = objects_srcs,
    include_dirs = include_dirs,
    libraries = ['boost_python', 'GL', 'GLU', 'core', '3ds']
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
        extra_dir = os.path.join(extra_dir, 'glpython')
        ext.library_dirs.append(extra_dir)
        build_ext.build_extension(self, ext)

class my_install_data(install_data):
    def finalize_options (self):
        self.set_undefined_options ('install', ('install_lib', 'install_dir'))
        install_data.finalize_options(self)

#______________________________________________________________________________
s = setup (name='glpython',
       version = 'beta',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description = "GLPython: OpenGL python terminal",
       ext_modules = [libcore_lib, core_ext,  libobjects_lib, objects_ext],       
       packages = ['glpython',
                   'glpython.core',
                   'glpython.objects',                   
                   'glpython.backends',
                   'glpython.terminal',
                   'glpython.data'],
       cmdclass = { 'install_data' : my_install_data,
                    'build_ext': my_build_ext},
       scripts=['bin/glpython', 'bin/glpython'],
       data_files = [
            ("glpython/data", glob.glob('glpython/data/*'))]
      )

print
print "======================================================================="
print
print " You have to modify your LD_LIBRARY_PATH environment variable by adding"
print "  $prefix + 'python%s/site-packages/glpython'" % sys.version[0:3]
print
print "======================================================================="
print

