#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------


import dana.core as core
import dana.projection as proj
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile

from dana.visualization import View2D
from glpython.window import window
from dana.gl.network import View as NView
from dana.gl.weights import View as WView

import numpy


w = 10
net = core.Network()

m = core.Map( (w,w), (0,0) )
m.name = "Map_1"
m.append( core.Layer() )
m[0].fill(core.Unit)
net.append(m)

m = core.Map( (w,w), (1,1) )
m.name = "Map_2"
m.append( core.Layer() )
m[0].fill(core.Unit)
net.append(m)

p = proj.projection()
p.distance = proj.distance.euclidean (True)
p.density = proj.density.full(1)
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.point()
p.src = m[0]
p.dst = m[0]
p.connect()


win = window(locals())
win.view.append (NView (net))
win.show()


