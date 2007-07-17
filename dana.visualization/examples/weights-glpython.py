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
# $Id: profile.py 119 2007-02-07 14:16:22Z rougier $
#------------------------------------------------------------------------------

import dana.core as core
import dana.projection as proj
from glpython import window, CM_IceAndFire
from dana.visualization.glpython import Figure

size = 10;
net = core.Network()
m0 = core.Map( (size, size), (0,0) )
m0.append( core.Layer() )
m0[0].fill(core.Unit)
m0.name = 'm0: constant'
net.append(m0)
m1 = core.Map( (size, size), (1,0) )
m1.append( core.Layer() )
m1[0].fill(core.Unit)
m1.name = 'm1: linear'
net.append(m1)
p          = proj.projection()
p.self     = True
p.distance = proj.distance.euclidean(False)
p.density  = proj.density.full(1)
p.shape    = proj.shape.box(1,1)
p.profile  = proj.profile.gaussian(1,.25)
p.src      = m0[0]
p.dst      = m1[0]
p.connect()

cmap = CM_IceAndFire
cmap.scale (-1,1)

fig = Figure()
win,fig = window (figure=fig)
fig.weights (m0[0], m1[0], title='Weights from m0 to m1', cmap=cmap)
win.show()

