#!/usr/bin/env python

# Some description
print "------------------------------------------------------------------------"
print "Selective attention using CNFT (full connectivity version)"
print ""
print "Authors:   Julien Vitay & Nicolas Rougier"
print "Date:      01/03/2005"
print "Reference: Vitay J. & Rougier N.P."
print '           "Using Neural Dynamics to Switch Attention"'
print "           International Joint Conference on Neural Networks, 2005."
print "------------------------------------------------------------------------"

# Import
import dana.core as core
import dana.cnft as cnft
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile
from glpython import window as glwindow
from dana.visualization.glpython import Figure

import time, random, math
import numpy

# Create a new network
model = core.Model()
net = core.Network ()
model.append(net)
width  = 40
height = 40

# Function for creating a new map of units (of type unit_class)
def new_map(name, w, h, px, py,unit_class):
    map = core.Map ((w,h), (px,py))
    map.name = name
    map.append(core.Layer())
    map[0].fill(unit_class)
    if(not(unit_class == core.Unit)):
	map.spec = cnft.Spec()
    net.append(map)
    return map

# Unit specification
print "Building maps"
# Maps
visual          = new_map ('visual', width, height, 0, 0, core.Unit)
focus          = new_map ('focus', width, height, 1, 2, cnft.Unit)
input         = new_map ('input', width, height, 0, 1, cnft.Unit)
wm            = new_map ('wm', width, height, 0, 2, cnft.Unit)
thal_wm       = new_map ('thal_wm', width, height, 0, 3, cnft.Unit)
inhibition     = new_map ('inhibition', width, height, 1, 1, cnft.Unit)
striatum_inhib = new_map ('striatum_inhib', width, height, 2, 3, cnft.Unit)
thal_inhib     = new_map ('thal_inhib', width, height, 2, 2, cnft.Unit)
gpi_inhib      = new_map ('gpi_inhib', width, height, 2, 1, cnft.Unit)
reward         = new_map ('reward', 1, 1, 2, 0, cnft.Unit)

p = projection.Projection() 
p.distance = distance.Euclidean (False)
p.density = density.Full(1)

# Visual connections
####################
print "Binding visual connections"

# visual to input
p.profile = profile.Gaussian(2.0,2.0/width)
p.shape = shape.Disc(2.5*2.0/width)
p.src = visual[0]
p.dst = input[0]
p.connect()

# input to focus
p.profile = profile.Gaussian(0.25,2.0/width)
p.shape = shape.Disc(2.5*2.0/width)
p.src = input[0]
p.dst = focus[0]
p.connect()

# input to wm
p.profile = profile.Gaussian(0.25,2.0/width)
p.shape = shape.Disc(2.5*2.0/width)
p.src = input[0]
p.dst = wm[0]
p.connect()

# CNFT connections
####################
print "Binding cnft connections"
# input
p.profile = profile.DoG(1.2,3.0/width,1.0,4.0/width)
p.shape = shape.Disc(2.5*4.0/width)
p.src = input[0]
p.dst = input[0]
p.connect()

# focus
p.profile = profile.DoG(1.7,4.0/width,0.65,1.0)
p.shape = shape.Disc(2.5*17.0/width)
p.src = focus[0]
p.dst = focus[0]
p.connect()

# wm
p.profile = profile.DoG(2.5,2.0/width,1.0,4.0/width)
p.shape = shape.Disc(2.5*4.0/width)
p.src = wm[0]
p.dst = wm[0]
p.connect()

# inhibition
p.profile = profile.DoG(2.5,2.0/width,1.0,4.0/width)
p.shape = shape.Disc(2.5*4.0/width)
p.src = inhibition[0]
p.dst = inhibition[0]
p.connect()

# CNFT on striatum_inhib
p.profile = profile.DoG(2.5,2.0/width,1.0,4.0/width)
p.shape = shape.Disc(2.5*4.0/width)
p.src = striatum_inhib[0]
p.dst = striatum_inhib[0]
p.connect()

# Inter-cortical connections
############################
print "Binding inter-cortical connections"

# focus to wm
p.profile = profile.Gaussian(0.2,2.0/width)
p.shape = shape.Disc(2.5*2.0/width)
p.src = focus[0]
p.dst = wm[0]
p.connect()

# fef to inhibition
p.profile = profile.Gaussian(0.25,2.5/width)
p.shape = shape.Disc(2.5*2.5/width)
p.src = wm[0]
p.dst = inhibition[0]
p.connect()

# inhibition to focus
p.profile = profile.Gaussian(-0.2,3.5/width)
p.shape = shape.Disc(2.5*3.5/width)
p.src = inhibition[0]
p.dst = focus[0]
p.connect()

# Working memory connections
############################
print "Binding working memory connections"

# wm to thal_wm
p.profile = profile.Gaussian(2.35,1.5/width)
p.shape = shape.Disc(2.5*1.5/width)
p.src = wm[0]
p.dst = thal_wm[0]
p.connect()

# thal_wm to wm
p.profile = profile.Gaussian(2.4,1.5/width)
p.shape = shape.Disc(2.5*1.5/width)
p.src = thal_wm[0]
p.dst = wm[0]
p.connect()

# Inhibition connections
########################
print "Binding inhibition connections"

# inhibition to thal_inhib
p.profile = profile.Gaussian(3.0,1.5/width)
p.shape = shape.Disc(1.5)
p.src = inhibition[0]
p.dst = thal_inhib[0]
p.connect()

# thal_inhib to inhibition
p.profile = profile.Gaussian(3.0,1.5/width)
p.shape = shape.Disc(1.5)
p.src = thal_inhib[0]
p.dst = inhibition[0]
p.connect()

# Basal ganglia connections
###########################
print "Binding basal ganglia connections"
# fef to striatum_inhib
p.profile = profile.Gaussian(0.5,2.5/width)
p.shape = shape.Disc(2.5*2.5/width)
p.src = wm[0]
p.dst = striatum_inhib[0]
p.connect()

# striatum_inhib to gpi_inhib
p.profile = profile.Gaussian(-2.5,2.5/width)
p.shape = shape.Disc(0.5)
p.src = striatum_inhib[0]
p.dst = gpi_inhib[0]
p.connect()

# gpi_inhib to thal_inhib
p.profile = profile.Gaussian(-1.5,1.5/width)
p.shape = shape.Disc(0.5)
p.src = gpi_inhib[0]
p.dst = thal_inhib[0]
p.connect()

# reward to striatum_inhib
p.profile = profile.Constant(8.0)
p.shape = shape.Box(1,1)
p.density = density.Full(1.0)
p.src = reward[0]
p.dst = striatum_inhib[0]
p.connect()

print "Links OK"

### Unit specs

input.spec.alpha = 23.0
input.spec.tau = 0.75

focus.spec.baseline = -0.05
focus.spec.alpha = 13.0

wm.spec.tau = 0.6
wm.spec.baseline = -0.25

inhibition.spec.baseline = -0.1

reward.spec.tau = 15

thal_wm.spec.tau = 0.6

striatum_inhib.spec.baseline = -0.5

gpi_inhib.spec.baseline = 1.0


# Show network
fig = Figure()
win,fig = glwindow (size=(800,600), title = "Selective Attention",has_terminal=True,namespace=locals(),figure=fig, fps=50)
fig.network (net, style='flat', show_colorbar=False, show_label=False)
fig.text (size=.1, position = (.5, -.05), text="Using Neural Dynamics to Switch Attention")
fig.text (size=.05, position = (.5, -.085), text="International Joint Conference on Neural Networks, 2005")
win.show()
