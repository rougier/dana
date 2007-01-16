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
print "Update : Overt Attention, Jeremy Fix"
print "------------------------------------------------------------------------"

# Import
import matplotlib.pylab as pylab
import matplotlib.colors as colors

import dana.core as core

import dana.sigmapi as sigmapi
import dana.sigmapi.projection as proj
import dana.sigmapi.projection.combination as combine

import dana.cnft as cnft

import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile

import dana.view as view

import time, random, math
import gobject, gtk

# Create a new network
net = core.Network ()
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
striatum_inhib.spec.baseline = -0.5
thal_inhib     = new_map ('thal_inhib', width/2, height/2, 2, 2, cnft.Unit)
gpi_inhib      = new_map ('gpi_inhib', width/2, height/2, 2, 1, cnft.Unit)
gpi_inhib.spec.baseline = 0.8
reward         = new_map ('reward', 1, 1, 2, 0, cnft.Unit)

# Anticipation related maps
anticipation = new_map ('anticipation',width,height,1,3, sigmapi.Unit)

p = projection.projection() 
p1 = proj.projection() # For sigmapi connections

# Visual connections
####################
print "Binding visual connections"

# visual to input
p.distance = distance.euclidean (False)
p.density = density.full(1)
p.profile = profile.gaussian(2.0,2.0/width)
p.shape = shape.disc(2.5*2.0/width)
p.src = visual[0]
p.dst = input[0]
p.connect()

# input to focus
p.profile = profile.gaussian(0.25,2.0/width)
p.shape = shape.disc(2.5*2.0/width)
p.src = input[0]
p.dst = focus[0]
p.connect()

# input to wm
p.profile = profile.gaussian(0.25,2.0/width)
p.shape = shape.disc(2.5*2.0/width)
p.src = input[0]
p.dst = wm[0]
p.connect()

# CNFT connections
####################
print "Binding cnft connections"
# input
p.profile = profile.dog(1.2,3.0/width,1.0,4.0/width)
p.shape = shape.disc(2.5*4.0/width)
p.src = input[0]
p.dst = input[0]
p.connect()

# focus
p.profile = profile.dog(1.7,4.0/width,0.65,17.0/width)
p.shape = shape.disc(2.5*17.0/width)
p.src = focus[0]
p.dst = focus[0]
p.connect()

# wm
p.profile = profile.dog(2.5,2.0/width,1.0,4.0/width)
p.shape = shape.disc(2.5*4.0/width)
p.src = wm[0]
p.dst = wm[0]
p.connect()

# inhibition
p.profile = profile.dog(2.5,2.0/width,1.0,4.0/width)
p.shape = shape.disc(2.5*4.0/width)
p.src = inhibition[0]
p.dst = inhibition[0]
p.connect()

# CNFT on striatum_inhib
p.profile = profile.dog(2.5,2.0/width,1.0,4.0/width)
p.shape = shape.disc(2.5*4.0/width)
p.src = striatum_inhib[0]
p.dst = striatum_inhib[0]
p.connect()

# Inter-cortical connections
############################
print "Binding inter-cortical connections"

# focus to wm
p.profile = profile.gaussian(0.2,2.0/width)
p.shape = shape.disc(2.5*2.0/width)
p.src = focus[0]
p.dst = wm[0]
p.connect()

# fef to inhibition
p.profile = profile.gaussian(0.25,2.5/width)
p.shape = shape.disc(2.5*2.5/width)
p.src = wm[0]
p.dst = inhibition[0]
p.connect()

# inhibition to focus
p.profile = profile.gaussian(-0.2,3.5/width)
p.shape = shape.disc(2.5*3.5/width)
p.src = inhibition[0]
p.dst = focus[0]
p.connect()

# Working memory connections
############################
print "Binding working memory connections"

# wm to thal_wm
p.profile = profile.gaussian(2.35,1.5/width)
p.shape = shape.disc(2.5*1.5/width)
p.src = wm[0]
p.dst = thal_wm[0]
p.connect()

# thal_wm to wm
p.profile = profile.gaussian(2.4,1.5/width)
p.shape = shape.disc(2.5*1.5/width)
p.src = thal_wm[0]
p.dst = wm[0]
p.connect()

# Inhibition connections
########################
print "Binding inhibition connections"

# inhibition to thal_inhib
p.profile = profile.gaussian(3.0,1.5/width)
p.shape = shape.disc(1.5)
p.src = inhibition[0]
p.dst = thal_inhib[0]
p.connect()

# thal_inhib to inhibition
p.profile = profile.gaussian(3.0,1.5/width)
p.shape = shape.disc(1.5)
p.src = thal_inhib[0]
p.dst = inhibition[0]
p.connect()

# Basal ganglia connections
###########################
print "Binding basal ganglia connections"
# fef to striatum_inhib
p.profile = profile.gaussian(0.5,2.5/width)
p.shape = shape.disc(2.5*2.5/width)
p.src = wm[0]
p.dst = striatum_inhib[0]
p.connect()

# striatum_inhib to gpi_inhib
p.profile = profile.gaussian(-2.5,2.5/width)
p.shape = shape.disc(0.5)
p.src = striatum_inhib[0]
p.dst = gpi_inhib[0]
p.connect()

# gpi_inhib to thal_inhib
p.profile = profile.gaussian(-1.5,1.0/width)
p.shape = shape.disc(0.5)
p.src = gpi_inhib[0]
p.dst = thal_inhib[0]
p.connect()

# reward_inhib to striatum_inhib
p.profile = profile.constant(8.0)
p.shape = shape.box(1,1)
p.density = density.full(1.0)
p.src = reward[0]
p.dst = striatum_inhib[0]
p.connect()

# Anticpation connections
#########################
print "Binding anticipation connections"

# wm -- focus -> anticipation
p1.combination = combine.linear(1,-1,1,1,-1,1,-width/2.0,-height/2.0,0.05)
p1.src1= wm[0]
p1.src2 = focus[0]
p1.dst = anticipation[0]
p1.connect();

## anticipation -> wm
p.profile = profile.gaussian(0.2,2.0/width)
p.shape = shape.disc(2.5*2.0/width)
p.src = anticipation[0]
p.dst = wm[0]
p.connect()

## anticipation <-> anticipation
p.profile = profile.dog(1.6,3.0/width,1.0,4.0/width)
p.shape = shape.disc(0.5)
p.src = anticipation[0]
p.dst = anticipation[0]
p.connect()

print "Links OK"

execfile("Parameters.py")

## Tools 

pan2do = 0.0
tilt2do = 0.0
radius = 4
nb_stim = 0

def clamp(map,x0,y0):
	global radius
	for u in map[0]:
		u.potential = u.potential +math.exp(-float((u.position[0]-x0)*(u.position[0]-x0)+(u.position[1]-y0)*(u.position[1]-y0))/float(radius*radius))
	return True

def clear(map):
	for u in map[0]:
		u.potential = 0
		
def clear_all():
	clear(visual)
	clear(focus)
	clear(input)
	clear(wm)
	clear(thal_wm)
	clear(anticipation)		
		
def decode(map):
    global pan2do
    global tilt2do
    pan2do = 0.0
    tilt2do = 0.0
    activite_totale = 0.0
    for u in map[0]:
	    x = u.position[0]
	    y = u.position[1]
            pan2do += map.unit(x,y).potential*(x-width/2)
            tilt2do += map.unit(x,y).potential*(y-height/2)
            activite_totale += map.unit(x,y).potential
    if(activite_totale!=0.0):
        pan2do /= activite_totale
        tilt2do /= activite_totale
    else:
        pan2do = 0.0
        tilt2do = 0.0
        print "[Warning] : decodage de la carte ",map.name," : activite totale nulle !!"

## Simulation 
is_init = 0
stim_array = 0

from numpy import *

def init(nb):
	global is_init,stim_array,nb_stim
	nb_stim = nb
	clear(visual)
	is_init = 1
	stim_array = zeros([nb,2],float)
	for i in range(nb):
		x0 = random.randint(0,width)
		y0 = random.randint(0,height)
		stim_array[i][0] = x0
		stim_array[i][1] = y0
		clamp(visual,x0,y0)

def move():
	global stim_array,nb_stim,pan2do,tilt2do
	decode(focus)
	for i in range(nb_stim):
		stim_array[i][0] = stim_array[i][0]-pan2do
		stim_array[i][1] = stim_array[i][1]-tilt2do

def refresh():
	global stim_array,nb_stim
	clear(visual)
	for i in range(nb_stim):
		clamp(visual,stim_array[i][0],stim_array[i][1])
	
def switch():
	clear(focus)	

def clear_all():
	clear(visual)
	clear(focus)
	clear(input)
	clear(wm)
	clear(thal_wm)
	clear(anticipation)

def net_init(widget,data=None):
	init(3)

def net_evaluate(widget,data=None):
	net.evaluate(3,False)

def net_move(widget,data=None):
	move()

def net_refresh(widget,data=None):
	refresh()
	
def net_clear(widget,data=None):
	clear_all()	
	
def net_switch(widget,data=None):
	for u in reward[0]:
		u.potential=1.0

window = gtk.Window(gtk.WINDOW_TOPLEVEL)
window.set_border_width(12)
vbox = gtk.VBox(True, 6)

init_button = gtk.Button("Init")
init_button.connect("clicked",net_init)
vbox.add(init_button)

step_button = gtk.Button("Step")
step_button.connect("clicked",net_evaluate)
vbox.add(step_button)

move_button = gtk.Button("Move")
move_button.connect("clicked",net_move)
vbox.add(move_button)

switch_button = gtk.Button("Switch")
switch_button.connect("clicked",net_switch)
vbox.add(switch_button)

refresh_button = gtk.Button("Refresh")
refresh_button.connect("clicked",net_refresh)
vbox.add(refresh_button)

clear_button = gtk.Button("Clear all")
clear_button.connect("clicked",net_clear)
vbox.add(clear_button)


window.add (vbox)
window.show_all()


# Show network
netview = view.view (net)

manager = pylab.get_current_fig_manager()

def updatefig(*args):
    netview.update()
    return True

gobject.idle_add(updatefig)
pylab.show()