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

import math, random
import dana.core as core
import dana.projection as proj

from glpython.window import window
from dana.visualization.gl.network import View

# Create a new network
net = core.Network ()
width  = 30
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
Focus[0].fill(core.Unit)
Focus.name = 'Focus'
net.append(Focus)


p = proj.projection()
p.distance = proj.distance.euclidean (True)
p.density = proj.density.full(1)
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.point()
p.src = Input[0]
p.dst = Focus[0]
p.connect()

p.self = False
p.profile = proj.profile.dog (2.20, 3.0/width, 0.65, 11.0/width)

p.shape = proj.shape.box(1,1)
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

win = window(locals(), backend='gtk')
win.view.append (View (net, fontsize=48))
win.show()

