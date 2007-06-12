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

include_dir = core[:core.find('python')]
include_dirs = [os.path.normpath (os.path.join (include_dir, '../include/dana/'))]
include_dirs.append (numpy.get_include())
include_dirs = include_dirs + pkgconfig('mirage')['include_dirs']

dana_image_srcs = glob.glob ("dana/image/*.cc")
dana_image_ext = Extension (
        'dana.image._image',
        sources = dana_image_srcs,
        include_dirs= include_dirs ,
        libraries = ['boost_python', 'boost_thread'] + pkgconfig('mirage')['libraries'],
        )

setup (name='dana.image',
       version = '1.0',
       author = 'Jeremy Fix',
       author_email = 'Jeremy.Fix@Loria.fr',
       url = 'http://www.loria.fr/~fix',
       description ="This packages provides a toolbox for processing an image from a file or a server",
       packages = ['dana.image'],
       ext_modules = [dana_image_ext],
       data_files= []
       )
