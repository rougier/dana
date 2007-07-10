#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Jeremy Fix.
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
import commands

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw



# Get core shared object filename
from dana.core._core import __file__ as core
from dana.projection._projection import __file__ as projection
from dana.cnft._cnft import __file__ as cnft

include_dir = core[:core.find('python')]
include_dirs = [os.path.normpath (os.path.join (include_dir, '../include/dana/'))]
include_dirs.append (numpy.get_include())

include_dir = cnft[:cnft.find('python')]
include_dirs.append ([os.path.normpath (os.path.join (include_dir, '../include/dana/'))])

include_dir = projection[:projection.find('python')]
include_dirs.append ([os.path.normpath (os.path.join (include_dir, '../include/dana/'))])

include_dirs = include_dirs +['/usr/include']

dana_svd_srcs = glob.glob ("dana/svd/*.cc")
dana_svd_ext = Extension (
        'dana.svd._svd',
        sources = dana_svd_srcs,
        include_dirs= include_dirs,
        libraries = ['boost_python', 'boost_thread']+ pkgconfig('gsl')['libraries'],
        extra_objects=[core,cnft,projection],
        )

setup (name='dana.svd',
       version = '1.0',
       author = 'Jeremy Fix',
       author_email = 'Jeremy.Fix@Loria.fr',
       url = 'http://my.website.fr',
       description ="SVD implementation",
       packages = ['dana.svd'],
       ext_modules = [dana_svd_ext],
       data_files= []
       )

