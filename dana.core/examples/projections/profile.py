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
import dana.projection as proj


size = 10
network = core.Network()
map = core.Map((10,10), (0,0))
network.append (map)
layer = core.Layer()
map.append (layer)
layer.fill (core.Unit)


p = proj.Projection()
p.self_connect = True
p.distance = proj.distance.Euclidean (False)
p.density  = proj.density.Full(1)
p.shape    = proj.shape.Box(1.0, 1.0)
p.profile  = proj.profile.Constant(1)
p.src      = layer
p.dst      = layer
p.connect()
print 'Constant profile'
print layer.unit(size/2,size/2).weights (layer)

for u in layer: u.clear()
p.profile = proj.profile.Linear (0, 1)
p.connect()
print 'Linear profile'
print layer.unit(size/2,size/2).weights (layer)

for u in layer: u.clear()
p.profile = proj.profile.Gaussian (1, .25)
p.connect()
print 'Gaussian profile'
print layer.unit(size/2,size/2).weights (layer)

for u in layer: u.clear()
p.profile = proj.profile.DoG (2.2,0.1, .65,.35)
p.connect()
print 'DoG profile'
print layer.unit(size/2,size/2).weights (layer)



