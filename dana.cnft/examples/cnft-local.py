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

import matplotlib.pylab as pylab
import matplotlib.colors as colors

import dana.core as core
import dana.projection as proj
import dana.cnft as cnft
from dana.visualization import View2D

import time, random, math
import gobject, gtk


print "------------------------------------------------------------------------"
print "CNFT using local connectivity"
print ""
print "Author:    Nicolas Rougier"
print "Date:      20/04/2005"
print "Reference: Rougier N.P. "
print '           "Dynamic Neural Field With Local Inhibition"'
print "           Biological Cybernetics, 94, 3, pp 169-179, March 2006."
print "------------------------------------------------------------------------"

# Create a new network
net = core.Network ()
width  = 40
height = width

# Create the input map
Input = core.Map ( (width,height), (0,0) )
Input.append(core.Layer())
Input[0].fill(core.Unit)
Input.name = 'Input'
net.append(Input)

# Create the focus map 
Focus = core.Map ( (width,height), (1,0) )
Focus.append (core.Layer())
Focus[0].fill(cnft.Unit)
Focus.name = 'Focus'

Focus.spec = cnft.Spec()
Focus.spec.tau      = 0.75
Focus.spec.baseline = 0.1
Focus.spec.alpha    = 12.5
Focus.spec.min_act  = -1.0
Focus.spec.max_act  =  1.0

net.append(Focus)

# Create input to focus connections
p = proj.projection()
p.distance = proj.distance.euclidean (True)
p.density  = proj.density.full(1)
p.profile  = proj.profile.constant(1.5)
p.shape    = proj.shape.point()
p.src = Input[0]
p.dst = Focus[0]
p.connect()

# Create focus laterals connections
p.self = False
#p.density = proj.density.sparser(.5)
p.profile = proj.profile.dog (3.15, 0.05, 0.9, 0.1)
p.shape = proj.shape.disc (8.0/width)
p.src = Focus[0]
p.dst = Focus[0]
p.connect()

for u in Input[0]:
    u.potential = random.uniform(0.0, 1.0)

for i in xrange(Input.shape[0]):
    for j in xrange(Input.shape[1]):
        x0 = i/float(Input.shape[0])-.25
        y0 = j/float(Input.shape[1])-.25
        x1 = i/float(Input.shape[0])-.75
        y1 = j/float(Input.shape[1])-.75
        Input[0].unit(i,j).potential =  + math.exp (-(x0*x0+y0*y0)/0.025) + math.exp (-(x1*x1+y1*y1)/0.025) + .05*random.uniform(-1.0, 1.0)
        
# Show network
netview = View2D (net)

manager = pylab.get_current_fig_manager()
cnt = 0
tstart = time.time()

def updatefig(*args):
#    global cnt, start,net
#    net.evaluate(1,False)
    netview.update()
#    cnt += 1
#    if cnt==500:
#        print 'FPS', cnt/(time.time() - tstart)
#        return False
    return True
cnt = 0

gobject.idle_add (updatefig)
pylab.show()


