#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006, Nicolas Rougier.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under the
# conditions described in the aforementioned license. The license is also
# available online at http://www.loria.fr/~rougier/pub/Licenses/BSD.txt
# 
# Author: Nicolas Rougier
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
   
