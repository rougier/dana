#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
# 
# $Id: shape.py 171 2007-05-28 18:30:43Z rougier $
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
p.shape    = proj.shape.Point()
p.profile  = proj.profile.Constant(1)
p.src      = layer
p.dst      = layer
p.connect()
print 'Point shape'
print layer.unit(size/2,size/2).weights (layer)


for u in layer: u.clear()
p.shape    = proj.shape.Box (.25, .25)
p.connect()
print 'Box shape'
print layer.unit(size/2,size/2).weights (layer)

for u in layer: u.clear()
p.shape    = proj.shape.Disc(.25)
p.connect()
print 'Disc shape'
print layer.unit(size/2,size/2).weights (layer)

for u in layer: u.clear()
p.shape    = proj.shape.Box (1.0, 0.0)
p.connect()
print 'Line shape'
print layer.unit(size/2,size/2).weights (layer)

for u in layer: u.clear()
p.shape    = proj.shape.Box (0.0, 1.0)
p.connect()
print 'Column shape'
print layer.unit(size/2,size/2).weights (layer)

for u in layer: u.clear()
p.shape    = proj.shape.Box (0.0, 1.0)
p.connect()
p.self_connect = False
p.shape    = proj.shape.Box (1.0, 0.0)
p.connect()
print 'Cross shape'
print layer.unit(size/2,size/2).weights (layer)


