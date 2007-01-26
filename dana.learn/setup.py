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
from dana.core._core import __file__ as core
from dana.cnft._cnft import __file__ as cnft

include_dir = core[:core.find('python')]
include_dirs = [os.path.normpath (os.path.join (include_dir, '../include/dana/'))]
include_dirs.append (numpy.get_include())

print '-----------------------------------------------------'
print 'Guessed include directory (based on package core) :'
print ' =>', include_dir
print '     If this is wrong, please modify setup.py'
print '-----------------------------------------------------'
print


learn_srcs = glob.glob ("dana/learn/*.cc")
learn_ext = Extension (
    'dana.learn._learn',
    sources = learn_srcs,
    libraries = ['boost_python'],
    include_dirs =  include_dirs,
    extra_objects=[core,cnft]
)


setup (name='dana.learn',
       version = '1.0',
       author = 'Jeremy Fix',
       author_email = 'Jeremy.Fix@loria.fr',
       url = 'http://www.loria.fr/~fix',
       description = "DANA: Learning rules",
       packages = ['dana.learn'],
       ext_modules = [learn_ext],
      )
