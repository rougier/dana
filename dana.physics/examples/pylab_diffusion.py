#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------

import matplotlib.pylab as pylab
import matplotlib.colors as colors
import dana.core as core
import dana.projection as proj
import dana.physics as physics
from dana.visualization.pylab.network import View
from dana.gui.gtk import ControlPanel
import time, random, math
import gobject, gtk

print "--------------------------------------------------------------------"
print "Diffusion demo"
print "Author:    Nicolas Rougier"
print "Date:      08/02/2007"
print "--------------------------------------------------------------------"

# Create a new model
model = core.Model()

# Create a new network
net = core.Network ()
model.append(net)
size  = 100

# Create the map
Map = core.Map ( (2*size,size), (0,0) )
Map.append(core.Layer())
Map[0].fill(physics.Particle)
net.append(Map)

# Create input to focus connections
p = proj.projection()
p.self = False
p.distance = proj.distance.euclidean (False)
p.density = proj.density.sparser(.1)
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.box(3.0/(2*size), 3.0/size)
p.src = Map[0]
p.dst = Map[0]
p.connect()


# Split map potentials in 2 parts
for u in Map[0]:
    if u.position[0] < Map.shape[0]/2:
        u.potential = 1
    else:
        u.potential = 0


# Network visualization and control (gtk)
view = View (net)

def updatefig(*args):
    view.update()
    return True

manager = pylab.get_current_fig_manager()
gobject.idle_add(updatefig)
panel = ControlPanel (model)
pylab.show()

