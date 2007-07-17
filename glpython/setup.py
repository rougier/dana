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
from distutils.util import get_platform
from distutils.command.install_data import install_data
import distutils.sysconfig
import numpy

include_dirs = []
include_dirs.append (numpy.get_include())


# HACK: we should get this file from distutils
libtmp_dir = "./build/lib.%s-%s/glpython/" % (get_platform(), sys.version[0:3])


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
    library_dirs = [libtmp_dir,],
    libraries = ['boost_python', 'GL', 'GLU', 'core']
)

libobjects_srcs = glob.glob ("glpython/objects/*.cc")
libobjects_srcs = filter(lambda x: not 'objects.cc' in x, libobjects_srcs)
libobjects_lib = Extension (
    'glpython.libobjects',
    sources = libobjects_srcs,
    include_dirs = include_dirs,
    library_dirs = [libtmp_dir,],
    libraries = ['GL', 'GLU', 'core']
)
objects_srcs = glob.glob ("glpython/objects/*.cc")
objects_ext = Extension (
    'glpython.objects._objects',
    sources = objects_srcs,
    include_dirs = include_dirs,
    library_dirs = [libtmp_dir,],
    libraries = ['boost_python', 'GL', 'GLU', 'core']
)


class my_install_data(install_data):
    def finalize_options (self):
        self.set_undefined_options ('install', ('install_lib', 'install_dir'))
        install_data.finalize_options(self)

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
       cmdclass = { 'install_data' : my_install_data },
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

