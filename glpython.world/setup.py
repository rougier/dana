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

# Get shared object filenames
from glpython.core._core import __file__ as glcore

include_dirs = []

include_dir = glcore[:glcore.find('python')]
include_dirs = [os.path.normpath (os.path.join (include_dir, '../include/'))]
include_dirs.append (numpy.get_include())


core_srcs = glob.glob ("glpython/world/core/*.cc")

core_ext = Extension (
    'glpython.world.core._core',
    sources = core_srcs,
    include_dirs = include_dirs + ['/usr/include/GL/'],
    library_dirs = [],
    libraries = ['boost_python','GL','GLU','X11'], # ,'Magick++','Magick'
    extra_objects=[glcore]
)

objects_srcs = glob.glob ("glpython/world/objects/*.cc")

objects_ext = Extension (
    'glpython.world.objects._objects',
    sources = objects_srcs,
    include_dirs = include_dirs + ['/usr/include/GL/'],
    library_dirs = [],
    libraries = ['boost_python','GL','GLU','X11'], # ,'Magick++','Magick'
    extra_objects=[glcore]
)

setup (name='glpython.world',
       version = '1.0',
       author = 'Jeremy Fix',
       author_email = 'Jeremy.Fix@loria.fr',
       url = 'http://www.loria.fr/~fix',
       description = "glpython.world : Environnement + robot",
       ext_modules = [core_ext, objects_ext],       
       packages = ['glpython.world','glpython.world.core', 'glpython.world.objects'],
       data_files = []
      )
