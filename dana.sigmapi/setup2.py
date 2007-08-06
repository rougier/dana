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
# $Id$
#------------------------------------------------------------------------------

# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


import glob
from distutils.core import setup, Extension
import distutils.sysconfig
import numpy

# Get core shared object filename
from dana.cnft._cnft import __file__ as cnft
from dana.core._core import __file__ as core
include_dir = core[:core.find('python')]
include_dir = os.path.normpath (os.path.join (include_dir, '../include/dana/'))

print '-----------------------------------------------------'
print 'Guessed include directory (based on package core) :'
print ' =>', include_dir
print '     If this is wrong, please modify setup.py'
print '-----------------------------------------------------'
print

from dana.sigmapi._sigmapi import __file__ as sigmapi

# Get core shared object filename
include_dir = core[:core.find('python')]
include_dir = os.path.normpath (os.path.join (include_dir, '../include/dana/'))

projection_srcs = glob.glob ("dana/sigmapi/projection/*.cc")
projection_ext = Extension (
    'dana.sigmapi.projection._projection',
    sources = projection_srcs,
    libraries = ['boost_python'],
    include_dirs =  [numpy.get_include(),include_dir],
    extra_objects=[core,cnft,sigmapi]
)

combination_srcs = glob.glob ("dana/sigmapi/projection/combination/*.cc")
combination_ext = Extension (
    'dana.sigmapi.projection.combination._combination',
    sources = combination_srcs,
    libraries = ['boost_python'],
    include_dirs =  [numpy.get_include(),include_dir],
    extra_objects=[core,cnft,sigmapi]
)

setup (name='dana.sigmapi',
       version = '1.0',
       author = 'Jeremy Fix',
       author_email = 'Jeremy.Fix@loria.fr',
       url = 'http://www.loria.fr/~fix',
       description = "DANA: Sigma Pi Neurons",
       packages = ['dana.sigmapi',
                   'dana.sigmapi.projection',
		   'dana.sigmapi.projection.combination'],
       ext_modules = [projection_ext,
            combination_ext],
       data_files= [("include/dana/sigmapi",glob.glob("dana/sigmapi/*.h")),
		    ("include/dana/sigmapi/projection",glob.glob("dana/sigmapi/projection/*.h")),
		    ("include/dana/sigmapi/projection/combination",glob.glob("dana/sigmapi/projection/combination/*.h"))]

      )
