#!/usr/bin/env python
#
# DANA --- Distributed Asyncronous Numerical Adaptive computing library
# Copyright (C) 2007 Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# $Id: __init__.py 162 2007-05-11 13:03:57Z rougier $

"""
DANA --- Distributed Asyncronous Numerical Adaptive computing package
=====================================================================

DANA is a multi-platform python library for distributed asynchronous
numerical and adaptive computation. The computational paradigm
supporting the library is grounded on the notion of a unit that is
essentially a potential that can vary along time under the influence
of some other units.

Available packages
------------------
"""

import os.path, pydoc
from tests import test

def _packages_info():
    packagedir = os.path.abspath (os.path.dirname(os.path.realpath(__file__)))
    doc = ""
    files = os.listdir (packagedir)
    for f in files:
        fullname = os.path.join(packagedir, f)
        if os.path.isdir(fullname) and not os.path.islink(fullname):
            init_file = os.path.join (fullname, "__init__.py")
            if os.path.exists (init_file):
                synopsis = pydoc.synopsis (init_file)
                if synopsis:
                    doc += f.ljust(16,' ') + "--- " + synopsis  + "\n"
                else:
                    doc += f.ljust(16,' ') + "\t\t--- N/A\n"
    return doc

__doc__ += _packages_info()
__doc__ += "\n\n"
   
