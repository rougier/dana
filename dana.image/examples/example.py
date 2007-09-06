#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Jeremy Fix.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------


# When the example is run, try :
#     init()
#     read("orientations.jpg")
#     clamp()

# If you then want to read an other image, try
#     read("new_image.jpg")
#     clamp()


# Import
import matplotlib.pylab as pylab
import matplotlib.colors as colors

import dana.core as core

import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile

from glpython.core import *
from glpython import window
from glpython.objects import *
from dana.visualization.glpython import Figure
from dana.gui.gtk import ControlPanel

import time, random, math
import gobject, gtk

#from glpython.window import window as glwindow
#from dana.gl.network import View

from dana.image import *

# Create a new network
model = core.Model()
net = core.Network ()
model.append(net)
width  = 100
height = 100

# Function for creating a new map of units (of type unit_class)
def new_map(name, w, h, p,unit_class):
    map = core.Map ((w,h), p) #(px,py,ox,oy,z)
    map.name = name
    map.append(core.Layer())
    map[0].fill(unit_class)
    if(not(unit_class == core.Unit)):
	map.spec = cnft.Spec()
    net.append(map)
    return map

print "Building maps"

I_intensity = new_map('Intensity', width, height, (0,0), core.Unit)
I_RG = new_map('RG', width, height, (1, 0), core.Unit)
I_GR = new_map('GR', width, height, (2, 0), core.Unit)
I_BY = new_map('BY', width, height, (3, 0), core.Unit)
I_YB = new_map('YB', width, height, (4, 0), core.Unit)
I_horiz = new_map('horiz', width, height, (0, 1), core.Unit)
I_vert = new_map('vert', width, height, (1, 1), core.Unit)
I_PI_4 = new_map('PI_4', width, height, (2, 1), core.Unit)
I_3PI_4 = new_map('3PI_4', width, height, (3, 1), core.Unit)

sal = Saliency(color=1,orientation=1,save=1,verbose=0)

def init():
    global sal
    sal.add_orientation(0)
    sal.add_orientation(math.pi/2.0)
    sal.add_orientation(math.pi/4.0)
    sal.add_orientation(3.0*math.pi/4.0)
    sal.set_map(0,I_intensity[0])
    sal.set_map(1,I_RG[0])
    sal.set_map(2,I_GR[0])
    sal.set_map(3,I_BY[0])
    sal.set_map(4,I_YB[0])
    sal.set_map(5,I_horiz[0])
    sal.set_map(6,I_vert[0])
    sal.set_map(7,I_PI_4[0])
    sal.set_map(8,I_3PI_4[0])

def read(img,type):
    global sal
    sal.read(img,type)


def clamp():
    global sal
    sal.process()
    sal.clamp()

def clear_all():
    for m in net:
        clear(m)

def clear(m):
    for u in m[0]:
        u.potential = 0.0

control = ControlPanel (model)
fig = Figure()
win,fig = window (figure=fig,has_terminal=True,namespace=locals())
fnet = fig.network (net, title='Dana.image sample script',show_label=False)
fnet.colorbar.cmap = CM_Fire

win.show()

