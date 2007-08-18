#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006,2007 Nicolas Rougier.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under the
# conditions described in the aforementioned license. The license is also
# available online at http://www.loria.fr/~rougier/pub/Licenses/BSD.txt
# 
# $Id: __init__.py 128 2007-02-08 09:53:28Z rougier $
#------------------------------------------------------------------------------
""" Physics simulation

"""

from dana.core import *
from _physics import *

__all__ = ['Particle']
