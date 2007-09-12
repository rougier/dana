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
import threading
import time
import dana.core as core
import dana.projection as proj
from glpython import window
from dana.visualization.glpython import Figure


# Create a new network
model = core.Model()
net = core.Network ()
model.append (net)
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
Focus.append(core.Layer())
Focus[0].fill(core.Unit)
Focus.name = 'Focus'
net.append(Focus)

proj.one_to_one (Input[0], Focus[0]).connect()
proj.dog (Focus[0], Focus[0],2.20, 3.0/width, 0.65, 11.0/width).connect()

for u in Input[0]:
    u.potential = random.uniform(0.0, 1.0)

for i in xrange(Input.shape[0]):
    for j in xrange(Input.shape[1]):
        x0 = i/float(Input.shape[0])-.25
        y0 = j/float(Input.shape[1])-.25
        x1 = i/float(Input.shape[0])-.75
        y1 = j/float(Input.shape[1])-.75
        Input[0].unit(i,j).potential =  + math.exp (-(x0*x0+y0*y0)/0.0125) + math.exp (-(x1*x1+y1*y1)/0.0125) + .15*random.uniform(0.0, 1.0)


fig = Figure()
win,fig = window (figure=fig, fps=0, has_terminal=True, namespace = locals())
fig.network (net, style = 'smooth', title='A simple network')

win.show()

