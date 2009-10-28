#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE
from distutils.core import setup
from dana import __version__

# ______________________________________________________________________________
long_description = '''
The computational  paradigm supporting  the DANA framework  is grounded  on the
notion of  a unit that is  a set of arbitrary  values that can  vary along time
under the influence of other units and learning. Each unit can be linked to any
other unit (including  itself) using a weighted link and  a group is structured
set of homogeneous  units.'''

# ______________________________________________________________________________
setup(name="dana",
      version=__version__,
      url='http://dana.loria.fr/',
#      download_url='http://dana.loria.fr/dist/',
      license='GPL',
      author='Nicolas Rougier',
      author_email='Nicolas.Rougier@loria.fr',
      description='Distributed Asynchronous Numerical Adaptive computing',
      long_description=long_description,
      classifiers=[
              'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Environment :: Web Environment',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Operating System :: OS Independent',
              'Programming Language :: C++',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              ],
      platforms='any',
      packages=['dana',
                'dana.tests',
                'dana.pylab']
)
