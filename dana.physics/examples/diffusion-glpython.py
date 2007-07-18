#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
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
import dana.physics as physics
from glpython import window, Colormap
from dana.visualization.glpython import Figure
import dana.gui.gtk as gui

# Create a new model
model = core.Model()

# Create a new network
net = core.Network ()
model.append(net)
size  = 64

# Create the map
Map = core.Map ( (2*size,size), (0,0) )
Map.append(core.Layer())
Map[0].fill(physics.Particle)
net.append(Map)

# Create input to focus connections
p = proj.projection()
p.self = False
p.distance = proj.distance.euclidean (False)
p.density = proj.density.full()
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.box(1.0/(2*size), 1.0/size)
p.src = Map[0]
p.dst = Map[0]
p.connect()

# Split map potentials in 2 parts and one barrier
for u in Map[0]:
    if u.position[0] < Map.shape[0]/2:
        u.potential = 1
    elif u.position[0] > Map.shape[0]/2:
        u.potential = 0
    elif (u.position[0] == Map.shape[0]/2  and
            (u.position[1] > 11*Map.shape[1]/20 or u.position[1] < 9*Map.shape[1]/20)):
        u.potential = -1
    else:
        u.potential = 0


cmap = Colormap()
cmap.append (0.0, (0,0,0))      # For the barrier
cmap.append (0.0001, (0,0,0))
cmap.append (0.00011, (1,1,1))
cmap.append (1.0, (1,0,0))

fig = Figure()
win,fig = window (backend='gtk', figure=fig)
fig.network (net, style='flat', cmap=cmap, show_colorbar=False)
fig.colorbar (cmap=cmap, orientation = 'horizontal', position = (.1,.05))
fig.text (size=.1, position = (.5, -.05), text="Particle diffusion demo")
fig.text (size=.05, position = (.5, -.075), text="Nicolas Rougier, 07/2007")
cmap.scale (-0.01,1.0)
panel = gui.ControlPanel (model)
win.show()

