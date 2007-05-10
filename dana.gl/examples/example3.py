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

import gtk
import random

import dana.core as core
import dana.projection as proj
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile
from glpython.window import window
from dana.gl.network import View

import numpy


w = 40
net = core.Network()

m1 = core.Map( (w,w), (0,0) )
m1.append( core.Layer() )
m1[0].fill(core.Unit)
net.append(m1)

m2 = core.Map( (w,w), (1,0) )
m2.append( core.Layer() )
m2[0].fill(core.Unit)
net.append(m2)


p = proj.projection()
p.distance = proj.distance.euclidean (True)
p.density = proj.density.full(1)
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.point()
p.src = m1[0]
p.dst = m2[0]
p.connect()


def push (widget=None):
    for u in m1[0]:
        u.potential = random.uniform(0,1)
    for u in m2[0]:
        u.potential = random.uniform(0,1)


win = gtk.Window ()
button = gtk.Button ('Push')
button.connect ('clicked', push)
win.add (button)
win.show_all()


win = window(locals(), backend='gtk')
win.view.append (View (net))
win.show()


