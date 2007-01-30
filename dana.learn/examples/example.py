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
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile
import time, random, math
import gobject, gtk
import numpy

execfile('weights.py')

# Create a new network
net = core.Network ()
width  = 10
height = width

# Create the input map
m0 = core.Map ( (width,height), (0,0) )
m0.append(core.Layer())
m0[0].fill(core.Unit)
m0.name = 'm0'
net.append(m0)

# Create the focus map 
m1 = core.Map ( (width,height), (1,0) )
m1.append (core.Layer())
m1[0].fill(learn.Unit)
m1.name = 'm1'
net.append(m1)

proj          = projection.projection()
proj.self     = True
proj.distance = distance.euclidean(False)
proj.density  = density.full(1)
proj.shape    = shape.box(1,1)
proj.profile  = profile.constant(1)
proj.src      = m0[0]
proj.dst      = m1[0]
proj.connect()





learner = learn.Learner()

# Hebb's rule
rule = numpy.array([[0,2],0,0,0,1,0])

# Oja's rule
#rule = numpy.array([0,0,0,0-1w²,1,0])

learner.add(m0[0],m1[0],rule)
learner.learn(0.1); # The parameter is the learning rate

# Show network
print "netview"
netview = view.network.NetworkView (net)
print "weightview"
weightsview = WeightsView(m1[0], m0[0])
#print "netview show"
#netview.show()
#print "weightsview show"
#weightsview.show()

manager = pylab.get_current_fig_manager()

def learn():
	learner.learn(1.0)

def updatefig(*args):
    netview.update()
    weightsview.update()
    return True
print "ok!"
gobject.idle_add (updatefig)
pylab.show()
print "ok!!"
