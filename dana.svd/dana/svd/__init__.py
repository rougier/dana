#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier - Jeremy Fix.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------
""" svdcore library

Packages used to optimize the computations performed in the links\n
This optimisation is based on Singular Value Decomposition \n
Ref : http://en.wikipedia.org/wiki/Singular_value_decomposition  \n

"""
from dana.core import *
from dana.projection import *
from dana.cnft import *


from _svd import *

__all__ = ['Unit','Link','Layer','Projection']


