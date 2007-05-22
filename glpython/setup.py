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


import glob
from distutils.core import setup
from distutils.core import setup, Extension
from distutils.command.install_data import install_data
import distutils.sysconfig
import numpy

include_dirs = []
include_dirs.append (numpy.get_include())


class my_install_data(install_data):
    def finalize_options (self):
        self.set_undefined_options ('install', ('install_lib', 'install_dir'))
        install_data.finalize_options(self)

setup (name='glpython',
       version = '1.1',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description = "GLPython: OpenGL python terminal",
       packages = ['glpython',
                   'glpython.backends',
                   'glpython.terminal',
                   'glpython.data'],
       cmdclass = { 'install_data' : my_install_data },
       scripts=['bin/glpython', 'bin/glpython'],
       data_files = [
            ("glpython/data", glob.glob('glpython/data/*'))]
      )

