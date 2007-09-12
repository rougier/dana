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
# $Id: __init__.py 74 2007-01-18 08:44:30Z fix $
#------------------------------------------------------------------------------

""" Dana::sigmapi::core objects. It overloads Unit and Link to provide the necessary tools to define and use sigmapi links.
"""

from _core import *

__all__ = ['Unit','Link']
