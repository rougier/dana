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
""" Distributed Asyncronous Numerical Adaptive computing

DANA is a multi-platform library for distributed asynchronous numerical and
adaptive computation. The computational paradigm supporting the library is
grounded on the notion of a unit that is essentially a potential that can
vary along time under the influence of some other units.

Available packages
------------------
"""

import os.path, pydoc
from test import test

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
   
