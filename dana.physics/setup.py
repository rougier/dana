#!/usr/bin/env python


# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


import glob
from distutils.core import setup, Extension
import distutils.sysconfig


# Get core shared object filename
from dana.core._core import __file__ as core
include_dir = core[:core.find('python')]
include_dir = os.path.normpath (os.path.join (include_dir, '../include/dana/'))

print '-----------------------------------------------------'
print 'Guessed include directory (based on package core) :'
print ' =>', include_dir
print '     If this is wrong, please modify setup.py'
print '-----------------------------------------------------'
print


physics_srcs = glob.glob ("dana/physics/*.cc")
physics_ext = Extension (
    'dana.physics._physics',
    sources = physics_srcs,
    libraries = ['boost_python'],
    include_dirs =  [include_dir],
    extra_objects=[core]
)


setup (name='dana.physics',
       version = '1.0',
       author = 'Nicolas Rougier',
       author_email = 'Nicolas.Rougier@loria.fr',
       url = 'http://www.loria.fr/~rougier',
       description = "DANA: Physics",
       packages = ['dana.physics'],
       ext_modules = [physics_ext],
      )
