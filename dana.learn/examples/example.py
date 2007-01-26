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
import dana.learn as learn
import dana.view as view
import time, random, math
import gobject, gtk
import numpy

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
Focus[0].fill(learn.Unit)
Focus.name = 'Focus'
net.append(Focus)

learner = learn.Learner()

rule = numpy.array([1.4,2.54,3,4],dtype=float)
rule.shape=[2,2]
learner.add(Input[0],Focus[0],rule)

learner.learn(1.0);

# Show network
netview = view.network.NetworkView (net)


manager = pylab.get_current_fig_manager()
cnt = 0
tstart = time.time()

def updatefig(*args):
    netview.update()
    return True
cnt = 0

gobject.idle_add (updatefig)
pylab.show()
