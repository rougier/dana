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
# $Id: life.py 179 2007-05-29 08:19:37Z rougier $
#------------------------------------------------------------------------------
#!/usr/bin/env python

import matplotlib.pylab as pylab
import dana.core as core
import dana.projection as proj
import dana.life as life
from dana.visualization.pylab.network import View
from dana.gui.gtk import ControlPanel
import time, random, math
import gobject, gtk

# Create a new network
model = core.Model()
net = core.Network ()
model.append(net)
size  = 100

# Create the map
Map = core.Map ( (size,size), (0,0) )
Map.append(core.Layer())
Map[0].fill(life.Unit)
net.append(Map)

# Create input to focus connections
p = proj.projection()
p.self = False
p.distance = proj.distance.euclidean (False)
p.density = proj.density.full(1)
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.box(1.0/size, 1.0/size)
p.src = Map[0]
p.dst = Map[0]
p.connect()


for u in Map[0]:
    u.potential = random.randint (0, 1)


# Network visualization and control (gtk)
view = View (net)

def updatefig(*args):
    view.update()
    return True

manager = pylab.get_current_fig_manager()
gobject.idle_add(updatefig)
panel = ControlPanel (model)
pylab.show()

