#!/usr/bin/env python

import matplotlib.pylab as pylab
import matplotlib.colors as colors

import dana.core as core
import dana.projection as proj
import dana.physics as physics
import netview

import time, random, math
import gobject, gtk


print "--------------------------------------------------------------------"
print "Diffusion demo"
print "Author:    Nicolas Rougier"
print "Date:      05/01/2007"
print "--------------------------------------------------------------------"
print ""


# Create a new network
net = core.Network ()
size  = 64

# Create the map
Map = core.Map ( (size,size), (0,0) )
Map.append(core.Layer())
Map[0].fill(physics.Unit)
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
    u.potential = random.uniform (-0.1, 0.1)

for i in range(10):
    index     = random.randint (0, size*size)
    Map[0].unit(index).source = True
    potential = 1* (2*random.randint (0,1)-1)
    Map[0].unit(index).potential = potential
    
        
# Show network
netview = netview.view (net)

manager = pylab.get_current_fig_manager()

def updatefig(*args):
    net.evaluate(1,False)
    netview.update()
    return True

gobject.idle_add(updatefig)
pylab.show()


