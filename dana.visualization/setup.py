#!/usr/bin/env python

#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: setup.py 145 2007-05-10 14:18:42Z rougier $
#------------------------------------------------------------------------------


# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


import glob
from distutils.core import setup
from distutils.core import setup, Extension
from distutils.command.install_data import install_data
import distutils.sysconfig
import numpy

include_dirs = []
include_dirs.append (numpy.get_include())

gl_srcs = glob.glob ("dana/visualization/gl/*.cc")
gl_ext = Extension (
    'dana.visualization.gl._gl',
    sources = gl_srcs,
    libraries = ['boost_python', 'GL', 'ftgl', 'freetype'],
    include_dirs =  include_dirs
)

setup (name='dana.visualization',
       version = '1.0',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description = "DANA: visualization tools",
       packages = ['dana.visualization',
                   'dana.visualization.pylab',
                   'dana.visualization.gl'],
       ext_modules = [gl_ext]
      )

