#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
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
from glpython import window
from dana.visualization.glpython import Figure
#import dana.gui.gtk as gui

model = core.Model()
net = core.Network ()
model.append(net)
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
Focus[0].fill(cnft.Unit)
Focus.name = 'Focus'

Focus.spec = cnft.Spec()
Focus.spec.tau      = 0.75
Focus.spec.baseline = 0.0
Focus.spec.alpha    = 13.5
Focus.spec.min_act  = 0.0
Focus.spec.max_act  = 1.0

net.append(Focus)

# Create input to focus connections
p = proj.Projection()
p.distance = proj.distance.Euclidean(True)#proj.distance.euclidean (True)
p.density = proj.density.Full(1)
p.profile = proj.profile.Constant(1.0)
p.shape = proj.shape.Point()
p.src = Input[0]
p.dst = Focus[0]
p.connect()

# Create focus laterals connections
(proj.dog (Focus[0],Focus[0],2.20, 3.0/width, 0.55, 1.0, False)).connect() # 11/width

env = cnft.Environment()
env.attach (Input)
model.append (env)

# Display the network
fig = Figure()
win,fig = window (backend='gtk', figure=fig, has_terminal=True, namespace=locals(), fps=50)
fig.network (net, style='flat', show_colorbar=False)
fig.text (size=.1, position = (.5, -.05), text="Emergence of Attention within a Neural Population")
fig.text (size=.05, position = (.5, -.085), text="Neural Networks, 19, 5, pp 573-581, June 2006")
#panel = gui.ControlPanel (model)
win.show()


