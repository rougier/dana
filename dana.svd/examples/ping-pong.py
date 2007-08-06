#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier - Jeremy Fix.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------

import random, math
import dana.core as core
import dana.projection as proj
import dana.cnft as cnft

from glpython.core import CM_Fire
from glpython import window
from dana.visualization.glpython import Figure
from dana.gui.gtk import ControlPanel

import dana.svd as svd

import time

print "-----------------------------------------------------------------------"
print " Ping-ping example"
print " The input map provides the targets, the focus map selects one of them,"
print " this map being inhibited by the inhib map which contains units with a "
print " longer time constant, providing a ping-pong effect in the Focus map   "
print " It works fine with 70x70 maps but the weights must be adjusted for    "
print " other sizes"
print "-----------------------------------------------------------------------"
print "(see the script for details)"

########
# See below to switch between non-optimized and optimized computations
########

# Create a new model
model = core.Model()
net = core.Network ()
model.append(net)
width  = 70
height = width

# Create the input map
Input = core.Map ( (width,height), (0,0) )
Input.append(core.Layer())
Input[0].fill(core.Unit)
Input.name = 'Input'
net.append(Input)

Noise = core.Map((width,height), (0,1))
Noise.append(core.Layer())
Noise[0].fill(core.Unit)
Noise.name = 'Noise'
net.append(Noise)

Inhib = core.Map ( (width,height), (1,1) )
Inhib.append(svd.Layer())
Inhib[0].fill(svd.Unit)
Inhib.name = 'Inhib'
net.append(Inhib)

Inhib.spec = cnft.Spec()
Inhib.spec.tau      = 20.0
Inhib.spec.baseline = 0.0
Inhib.spec.alpha    = 8.0
Inhib.spec.min_act  = 0.0
Inhib.spec.max_act  = 1.0
Inhib.spec.wp = 1
Inhib.spec.wm = 1


# Create the focus map 
Focus = core.Map ( (width,height), (1,0) )
Focus.append(svd.Layer())
Focus[0].fill(svd.Unit)
Focus.name = 'Focus'

Focus.spec = cnft.Spec()
Focus.spec.tau      = 10.0
Focus.spec.baseline = 0.0
Focus.spec.alpha    = 8.0
Focus.spec.min_act  = 0.0
Focus.spec.max_act  = 1.0
Focus.spec.wp = 1
Focus.spec.wm = 1

net.append(Focus)

# Create input to focus connections
p1 = svd.projection()

# We propose different types of links :
# p1.separable = 0 : core::Link
# p1.separable = 1 : svd::Link the links are shared and contained by the layer
# p1.separable = 2 : svd::Link computed with Singular Value Decomposition

p1.self = True
p1.separable = 0
p1.distance = proj.distance.euclidean(True)
p1.profile = proj.profile.constant(1.0)
p1.density = proj.density.full(1)
p1.shape = proj.shape.point()
p1.src = Noise[0]
p1.dst = Focus[0]
p1.connect()

p1.separable = 2
p1.distance = proj.distance.euclidean (True)
p1.profile = proj.profile.gaussian(0.5,0.05)
p1.density = proj.density.full(1)
p1.shape = proj.shape.disc(1)
p1.src = Input[0]
p1.dst = Focus[0]
p1.connect()

p1.separable = 2
p1.distance = proj.distance.euclidean (True)
p1.profile = proj.profile.gaussian(0.5,0.05)
p1.density = proj.density.full(1)
p1.shape = proj.shape.disc(1)
p1.src = Focus[0]
p1.dst = Inhib[0]
p1.connect()

p1.separable = 2
p1.distance = proj.distance.euclidean (True)
p1.profile = proj.profile.gaussian(-0.6,0.05)
p1.density = proj.density.full(1)
p1.shape = proj.shape.disc(1)
p1.src = Inhib[0]
p1.dst = Focus[0]
p1.connect()

p1.separable = 2
p1.distance = proj.distance.euclidean (True)
p1.profile =  proj.profile.dog(0.8,0.1,0.6,1.4)
p1.density = proj.density.full(1)
p1.shape = proj.shape.box(1,1)
p1.src = Focus[0]
p1.dst = Focus[0]
p1.connect()

for u in Input[0]:
    u.potential = random.uniform(0.0, 1.0)

radius = 0.1

for i in xrange(Input.shape[0]):
    for j in xrange(Input.shape[1]):
        x0 = i/float(Input.shape[0])-.25
        y0 = j/float(Input.shape[1])-.25
        x1 = i/float(Input.shape[0])-.75
        y1 = j/float(Input.shape[1])-.75 
        Input[0].unit(i,j).potential =  + math.exp (-(x0*x0+y0*y0)/(radius*radius)) + math.exp (-(x1*x1+y1*y1)/(radius*radius))
        Noise[0].unit(i,j).potential =  .15*random.uniform(0.0, 1.0)
        

def evaluate(nb_steps):
    start = time.time()
    net.evaluate(nb_steps, False)
    end = time.time()
    print 'Elapsed time : %f second(s)' % (end-start)    

# Show network
fig = Figure()
win,fig = window (figure=fig,has_terminal=True,namespace=locals())
fignet = fig.network (net, style = 'flat', title='Ping pong')
fignet.colorbar.cmap = CM_Fire
control = ControlPanel (model)
win.show()
