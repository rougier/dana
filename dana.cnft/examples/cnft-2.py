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

import random, math
import dana.core as core
import dana.projection as proj
import dana.cnft as cnft
from glpython.window import window
from dana.visualization.gl.network import View
from dana.gui.gtk import ControlPanel


print "--------------------------------------------------------------------"
print "CNFT using full connectivity"
print ""
print "Author:    Nicolas Rougier"
print "Date:      01/03/2005"
print "Reference: Rougier N.P. & Vitay J."
print "           'Emergence of Attention within a Neural Population'"
print "           Neural Networks, 19, 5, pp 573-581, June 2006."
print "--------------------------------------------------------------------"
print ""


# Create a new model
model = core.Model()
net = core.Network ()
model.append(net)
width  = 50
height = width

# Create the input map
Input = core.Map ( (width,height), (0,0) )
Input.append(core.Layer())
Input[0].fill(core.Unit)
Input.name = 'Input'
net.append(Input)

# Create the focus map 
Focus = core.Map ( (width,height), (1,0) )
Focus.append(core.Layer())
Focus[0].fill(cnft.KUnit)
Focus.name = 'Focus'

Focus.spec = cnft.Spec()
Focus.spec.tau      = 0.75
Focus.spec.baseline = 0.0
Focus.spec.alpha    = 13.0
Focus.spec.min_act  = 0.0
Focus.spec.max_act  = 1.0
Focus.spec.wp = 1
Focus.spec.wm = 1

net.append(Focus)

# Create input to focus connections
p = proj.projection()
p.distance = proj.distance.euclidean (True)
p.density = proj.density.full(1)
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.point()
p.src = Input[0]
p.dst = Focus[0]
p.connect()

# Create focus laterals connections
p.self = False
p.density = proj.density.sparser(.85)
p.profile = proj.profile.uniform (-0.25, 0)
p.shape = proj.shape.disc(1)
p.src = Focus[0]
p.dst = Focus[0]
p.connect()

p.profile = proj.profile.uniform (0, .45)
p.shape = proj.shape.disc(.2)
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
        Input[0].unit(i,j).potential =  + math.exp (-(x0*x0+y0*y0)/0.0125) + math.exp (-(x1*x1+y1*y1)/0.0125) + .15*random.uniform(0.0, 1.0)
        

# Show network
win = window(locals(), backend='gtk')
win.view.append (View (net, fontsize=48))
control = ControlPanel (model)
win.show()

