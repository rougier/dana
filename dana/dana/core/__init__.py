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

From a conceptual point of view, the computational paradigm supporting the
library is grounded on the notion of a unit that is essentially a potential
that can vary along time under the influence of other units and learning.
Those units are organized into layers, maps and network: a network is made
of one to several map, a map is made of one to several layer and a layer is
made of a set of several units. Each unit can be linked to any other unit
(included itself) using a weighted link.

"""

from _core import *

__all__ = ['Model', 'Environment', 'Network', 'Map', 'Layer', 'Unit', 'Link']
