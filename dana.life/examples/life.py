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
#!/usr/bin/env python

import matplotlib.pylab as pylab
import matplotlib.colors as colors

import dana.core as core
import dana.projection as proj
import dana.life as life
import dana.view as view

import time, random, math
import gobject, gtk


print "--------------------------------------------------------------------"
print "Game Of Life demo"
print "Author:    Nicolas Rougier"
print "Date:      05/01/2007"
print ""
print "Comment: This package demonstrates how to cheat D.A.N.A. to peform"
print "         synchronous computation using compute_dw function."
print "--------------------------------------------------------------------"
print ""


# Create a new network
net = core.Network ()
size  = 200

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

        
# Show network
netview = view.view (net)

manager = pylab.get_current_fig_manager()

def updatefig(*args):
    net.evaluate(1,False)
    netview.update()
    return True

gobject.idle_add(updatefig)
pylab.show()


