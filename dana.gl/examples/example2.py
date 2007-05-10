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
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile

from glpython.window import window
from dana.gl.weights import View


net = core.Network()

m0 = core.Map( (20,20), (0,0) )
m0.append( core.Layer() )
m0[0].fill(core.Unit)
net.append(m0)

m1 = core.Map( (20,20), (0,0) )
m1.append( core.Layer() )
m1[0].fill(core.Unit)
net.append(m1)

proj = projection.projection()
proj.self = False
proj.distance = distance.euclidean(False)
proj.density = density.full(1)
proj.shape = shape.box(1,1)
proj.profile = profile.uniform(0.0, 1.0)
proj.src = m1[0]
proj.dst = m0[0]
proj.connect()

win = window()
win.view.append (View (layer=m0[0], source=m1[0]))
win.show()

