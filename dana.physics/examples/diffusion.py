#!/usr/bin/env python

import dana.core as core
import dana.projection as proj
import dana.physics as physics
from dana.visualization import View2D
import time, random, math
import gobject, gtk


print "--------------------------------------------------------------------"
print "Diffusion demo"
print "Author:    Nicolas Rougier"
print "Date:      08/02/2007"
print "--------------------------------------------------------------------"

# Create a new network
net = core.Network ()
size  = 50

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

        
# Show network
view = View2D (net)

def updatefig(*args):
    net.evaluate(1)
    view.update()
    return True

gobject.idle_add(updatefig)
view.show()


