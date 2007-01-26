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
import dana.view as view
import time, random, math
import gobject, gtk


# Create a new network
net = core.Network ()
width  = 30
height = width

# Create the input map
Input = core.Map ( (20,20), (0,0) )
Input.append(core.Layer())
Input[0].fill(core.Unit)
Input.name = 'Input'
net.append(Input)

# Create the focus map 
Focus = core.Map ( (width,height), (1,0) )
Focus.append (core.Layer())
Focus[0].fill(cnft.LUnit)
Focus.name = 'Focus'

Focus.spec = cnft.Spec()
Focus.spec.tau      = 1.5
Focus.spec.baseline = 0.1
Focus.spec.alpha    = 12.5
Focus.spec.min_act  = -1.0
Focus.spec.max_act  =  1.0
Focus.spec.lrate = .01
Focus.spec.wp = .5
Focus.spec.wm = .5

net.append(Focus)

# Create input to focus connections
p = proj.projection()
p.distance = proj.distance.euclidean (True)
p.density  = proj.density.sparse (.75)
p.profile  = proj.profile.uniform(0,.1)
p.shape    = proj.shape.disc(1)
p.src = Input[0]
p.dst = Focus[0]
p.connect()

# Create focus laterals connections
p.self = False
p.density = proj.density.sparser(.5)
p.profile = proj.profile.dog (3.15, .1, 0.9, .75)
p.shape = proj.shape.disc (1)
p.src = Focus[0]
p.dst = Focus[0]
p.connect()



def bubble():
    x0 = random.randint(-1,1) * .25
    y0 = random.randint(-1,1) * .25    
    for i in xrange(Input.shape[0]):
        for j in xrange(Input.shape[1]):
            x = i/float(Input.shape[0])-.5 +x0
            y = j/float(Input.shape[1])-.5 +y0
            Input[0].unit(i,j).potential = math.exp (-(x*x+y*y)/0.025)

        
# Show network
netview = view.network.NetworkView (net)


def run(e,n):
    for j in range(e):
        net.clear()
        bubble()
        for i in range(n):
            net.evaluate(1)
#            netview.update()



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


