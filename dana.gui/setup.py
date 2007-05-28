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

class my_install_data(install_data):
    def finalize_options (self):
        self.set_undefined_options ('install', ('install_lib', 'install_dir'))
        install_data.finalize_options(self)

setup (name='dana.gui',
       version = '1.0',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description = "DANA: GUI frontends",
       packages = ['dana.gui',
                   'dana.gui.data',
                   'dana.gui.gtk'],
       cmdclass = { 'install_data' : my_install_data },
       data_files = [("dana/gui/data", glob.glob('dana/gui/data/*.glade'))]
      )

