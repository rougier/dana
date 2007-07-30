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
import distutils.sysconfig
import numpy

# Get core shared object filename
from dana.core._core import __file__ as core
include_dir = core[:core.find('python')]
include_dirs = [os.path.normpath (os.path.join (include_dir, '../include/dana/'))]
include_dirs.append (numpy.get_include())

print '-----------------------------------------------------'
print 'Guessed include directory (based on package core) :'
print ' =>', include_dir
print '     If this is wrong, please modify setup.py'
print '-----------------------------------------------------'
print


cnft_srcs = glob.glob ("dana/cnft/*.cc")
cnft_ext = Extension (
    'dana.cnft._cnft',
    sources = cnft_srcs,
    libraries = ['boost_python'],
    include_dirs =  include_dirs,
    extra_objects=[core]
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
    def initialize_options(self):
        self.package = None
        build_ext.initialize_options(self)
		
    def finalize_options(self):
        if self.package is None:
            self.package = self.distribution.ext_package
        build_ext.finalize_options(self)
		
    def build_extension(self, ext):
        force_optimisation(self.compiler)
        extra_dir = self.build_lib
        if self.package:
            extra_dir = os.path.join(extra_dir, self.package)
        ext.library_dirs.append(extra_dir)
        build_ext.build_extension(self, ext)

#______________________________________________________________________________
setup (name='dana.cnft',
       version = '1.0',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description = "DANA: Continuum Neural Field Theory",
       packages = ['dana.cnft'],
       ext_modules = [cnft_ext],
       cmdclass = {
		"build_ext": my_build_ext
	   },
       data_files = [
            ("include/dana/cnft", glob.glob("dana/cnft/*.h")) ]
      )
