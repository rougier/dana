#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
# 
# $Id$
#-------------------------------------------------------------------------------

import dana.core as core
import dana.projection as projection

size = 10
network = core.Network()
map = core.Map((10,10), (0,0))
network.append (map)
layer = core.Layer()
map.append (layer)
layer.fill (core.Unit)

p = projection.dog (layer, layer, a1=1, b1=.5, a2=.5, b2=.75, self_connect=True)
p.connect()

print "Weights of unit[%d,%d]:" % (size/2, size/2)
print layer.unit(size/2,size/2).weights (layer)
