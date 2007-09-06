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
from distutils.command.build_ext import build_ext
from distutils.command.install_data import install_data
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
        #extra_dir = os.path.join(extra_dir, 'dana')
        ext.library_dirs.append(extra_dir)
        build_ext.build_extension(self, ext)

class my_install_data(install_data):
    def finalize_options (self):
        self.set_undefined_options ('install', ('install_lib', 'install_dir'))
        install_data.finalize_options(self)

#______________________________________________________________________________



setup (name='dana.image',
       version = '1.0',
       author = 'Jeremy Fix',
       author_email = 'Jeremy.Fix@Loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description ="This packages provides a toolbox for processing an image from a file or a server",
       packages = ['dana.image'],
       ext_modules = [dana_image_ext],
       cmdclass = { 'install_data' : my_install_data,
                    'build_ext': my_build_ext},       
       data_files= []
       )
