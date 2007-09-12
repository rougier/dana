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

import numpy
#from glpython import *
from glpython.world.core import *
from glpython.world.objects import *

# Create a new network
model = core.Model()
net = core.Network ()
model.append(net)
width  = 100
height = 100

# Create the robot
roger = Robot()

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

def read():
    global sal
    roger.grab("source.ppm")
    sal.read("source.ppm",'ppm')
    clamp()

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

# Overload of the key_press method of window
# to grab the key press event in the child viewports
def key_press_perso (self,key):
    """ """
    if key == 'control-d':
        if win.terminal and self.terminal.rl.line == '':
            win.destroy()
    elif key == 'control-f' or key == 'f11':
        if win.fullscreened:
            win.unfullscreen()
            win.fullscreened = False
        else:
            win.fullscreen()
            win.fullscreened = True
    elif key in ['f1', 'f2', 'f3', 'f4'] and win.terminal:
        layouts = {'f1': 1, 'f2':2, 'f3':3, 'f4':4}
        win.set_layout (layouts[key])
    else:
        taille = len(self.figure)

        for i in range(taille):
            o = self.figure.__getitem__(i)
            if(hasattr(o,"has_focus")):
                # The object is a viewport
                if(o.has_focus):
                    # The object has the focus
                    o.key_press_event(key);
                    return
        # Si j'arrive ici , c'est qu'aucun viewport n'a le focus
        if win.terminal:
            win.terminal.key_press(key)

control = ControlPanel (model)
fig = Figure()
win,fig = window (figure=fig,has_terminal=True,namespace=locals())
win.__class__.key_press = key_press_perso
fnet = fig.network (net, title='glpython.world sample script',show_colorbar=False, show_label = False)

# Define the view at third person
fig1 = Viewport(position=(-1,-1),size = (.25,.25))

# Set the view of the robot
roger.view.position = (-1,0.5)
roger.view.size = (.25,.25)

# Append the 3rd person view
# and the view of the robot
fig.append(roger.view)
fig.append(fig1)

# Append the robot to the scene
fig1.append(roger)

# Append a yellow disc to the scene
# pointing toward the robot
d = Disc()
d.phi = 90
d.radius = 0.1
d.z = 1.0
d.color[0] = 1
d.color[1] = 1
d.color[2] = 0
fig1.append(d)
roger.view.append(d)

# Append bars to the scene

# A 45 degrees, green bar, at (1,0)
gb = Bar()
gb.color[0] = 0
gb.color[1] = 1
gb.color[2] = 0
gb.theta = 45
gb.y = 1
gb.z = 0
fig1.append(gb)
roger.view.append(gb)

# A 90 degrees, blue bar, at (0,0)
bb = Bar()
bb.color[0] = 0
bb.color[1] = 0
bb.color[2] = 1
bb.theta = 90
bb.y = 0
bb.z = 0
fig1.append(bb)
roger.view.append(bb)

# A 135 degrees, red bar, at (-1,0)
rb = Bar()
rb.color[0] = 1
rb.color[1] = 0
rb.color[2] = 0
rb.theta = 135
rb.y = -1
rb.z = 0
fig1.append(rb)
roger.view.append(rb)


cmap = Colormap()
cmap.append (0, (0,0,0))
cmap.append (1, (0,0,0))
bg = Background (cmap)
roger.view.append(bg)
fig1.append(bg)

win.show()


