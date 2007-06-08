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
include_dir = []
include_dir.append(os.path.normpath ('/users/cortex/fix/local/include/jpeg-c++-1.0'))
include_dir.append(os.path.normpath ('/users/cortex/fix/local/include/mirage-1.0'))


dana_image_srcs = glob.glob ("dana/image/*.cc")
dana_image_ext = Extension (
        'dana.image._image',
        sources = dana_image_srcs,
        include_dirs=[numpy.get_include()] + include_dir,
        library_dirs=['/users/cortex/fix/local/lib/'],
        libraries = ['boost_python', 'boost_thread','m','mirage','jpeg' ]
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
