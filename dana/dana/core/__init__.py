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
""" core library

The core library implements the distributed asynchronous numerical and
adaptive paradigm. The base object is the unit object which is essentially
a potential that vary along time under the influence of other units
throughout link objects. Those units are gathered in a layer and several
layers (that share a common shape) compose what is called a map. A network
is a set of one to several maps.

The environment class describes some autonomous process that is able to
directly modify map activities.

Finally, a network and one to several environment(s) are gathered within a
simulation that can be ran and evaluated.
"""

from _core import *

__all__ = ['Model', 'Environment', 'Network', 'Map', 'Layer', 'Unit', 'Link']
