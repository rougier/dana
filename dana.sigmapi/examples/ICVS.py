#!/usr/bin/env python

print "--------------------------------------"
print "-         Hamker Demo                -"
print "-  A dynamic model of how feature    -"
print "-   cues guide spatial attention     -"
print "--------------------------------------"
print " Jeremy Fix                20/12/2005 "
print "--------------------------------------"

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
width  = 30
height = 30
width2 = 1
height2 = 1

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

print "Building maps"

# Maps F. Hamker
I_blue = new_map('I Blue', width, height, 2, 0, core.Unit)
I_green = new_map('I Green', width, height, 3, 0, core.Unit)
I_PI_4 = new_map('I Pi/4', width, height, 4, 0, core.Unit)
I_3PI_4 = new_map('I 3PI/4', width, height, 5, 0, core.Unit)

V4_blue = new_map('V4 Blue', width, height, 2, 1, sigmapi.Unit)
V4_green = new_map('V4 Green', width, height, 3, 1, sigmapi.Unit)
V4_PI_4 = new_map('V4 Pi/4', width, height, 4, 1, sigmapi.Unit)
V4_3PI_4 = new_map('V4 3PI/4', width, height, 5, 1, sigmapi.Unit)

IT_blue = new_map('IT Blue',width2, height2, 0, 2, sigmapi.Unit)
IT_green = new_map('IT Green', width2, height2, 1, 2, sigmapi.Unit)
IT_PI_4 = new_map('IT Pi/4', width2, height2, 2, 2, sigmapi.Unit)
IT_3PI_4 = new_map('IT 3PI/4', width2, height2, 3, 2, sigmapi.Unit)

ITt_blue = new_map('ITt Blue',width2, height2, 0, 3, sigmapi.Unit)
ITt_green = new_map('ITt Green', width2, height2, 1, 3, sigmapi.Unit)
ITt_PI_4 = new_map('ITt Pi/4', width2, height2, 2, 3, sigmapi.Unit)
ITt_3PI_4 = new_map('ITt 3PI/4', width2, height2, 3, 3, sigmapi.Unit)

PF_blue = new_map('PF Blue', 1, 1, 0, 4, core.Unit)
PF_green = new_map('PF Green', 1, 1, 1, 4, core.Unit)
PF_PI_4 = new_map('PF Pi/4', 1, 1, 2, 4, core.Unit)
PF_3PI_4 = new_map('PF 3Pi/4', 1, 1, 3, 4, core.Unit)

ff = new_map('ff', 1,1,0,0,core.Unit)
ff.unit(0).potential = 0.25

V4_to_percept = new_map('V4_to_percept', 1, 1, 1, 0, core.Unit)
V4_to_percept.unit(0).potential = 1.0

percept = new_map('Perceptual', width, height, 5, 2, sigmapi.Unit)
premotor = new_map('Premotor', width, height, 6, 2, cnft.Unit)

# Maps J. Vitay
width2 = 30
height2 = 30
wm            = new_map ('wm', width, height, 5, 3, cnft.Unit)
#wm_mod = new_map('wm_mod', 1, 1, 1, 0, core.Unit)
#wm_mod.unit(0).potential = 1.0

thal_wm       = new_map ('thal_wm', width, height, 5, 4, cnft.Unit)
inhibition     = new_map ('inhibition', width, height, 6, 3, cnft.Unit)
striatum_inhib = new_map ('striatum_inhib', width, height, 7, 3, cnft.Unit)
thal_inhib     = new_map ('thal_inhib', width2,height2 , 6, 4, cnft.Unit)
gpi_inhib      = new_map ('gpi_inhib', width2, height2, 7, 4, cnft.Unit)
switch         = new_map ('switch', 1, 1, 8, 4, cnft.Unit)

anticipation = new_map ('anticipation',width,height,3,3, sigmapi.Unit)
panticipation = new_map ('post_anticipation',width,height,4,3, sigmapi.Unit)
print "Maps built"

# Links 
p = projection.projection() 
p.self     = True
p.distance = distance.euclidean(False)
p.density  = density.full(1)

p1 = proj.projection() # For sigmapi connections

features = ['blue','green','PI_4','3PI_4']

### Feedforward Image -> V4
print "Connecting retina to V4"

for name in ['blue','green','PI_4','3PI_4']:
	exec("p1.src1 = I_"+name+"[0]")
	p1.src2 = ff[0]
	exec("p1.dst = V4_"+name+"[0]")
	p1.connect_point_mod_one(1.0)


#####################
######## What Pathway

### Feedforward : V4 -> IT
print "Connecting V4 to IT"

for name in ['blue','green','PI_4','3PI_4']:
	exec("p1.src1 = V4_"+name+"[0]")
	exec("p1.dst = IT_"+name+"[0]")
	p1.connect_all_to_one(1.0)


#### PF enhances the sensitivity of ITt neurons
print "Connecting IT---PF-->ITt"

for name in ['blue','green','PI_4','3PI_4']:
	exec("p1.src1 = IT_"+name+"[0]")
	exec("p1.src2 = PF_"+name+"[0]")
	exec("p1.dst = ITt_"+name+"[0]")
	p1.connect_point_mod_one(1.0)

### Feedback : IT -> V4

print "Connecting I---ITt--->V4"

for name in ['blue','green','PI_4','3PI_4']:
	exec("p1.src1 = I_"+name+"[0]")
	exec("p1.src2 = ITt_"+name+"[0]")
	exec("p1.dst = V4_"+name+"[0]")
	p1.connect_point_mod_one(0.5)


######################
######## Where Pathway

### Feedforward V4 -> Perceptual map

print "Where pathway : Connecting V4 to percept"

for name in ['blue','green','PI_4','3PI_4']:
	exec("p1.src1 = V4_"+name+"[0]")
	p1.src2 = V4_to_percept[0]
	p1.dst = percept[0]
	p1.connect_point_mod_one(1.0)

### Feedforward Perceptual -> Premotor map

print "Connecting perceptual to premotor"
#l.connect_as_dog(2,1.3,17,0.1)
p.profile = profile.dog(1.3,2.0/width,0.1,17.0/width)
p.shape = shape.disc(2.5*17.0/width)
p.src = percept[0]
p.dst = premotor[0]
p.connect()


#### CNFT 
print "Binding premotor cnft connections"
#l.connect_as_dog(5,0.95,17,0.65)
p.profile = profile.dog(0.95,5.0/width,0.65,17.0/width)
p.shape = shape.disc(2.5*17.0/width)
p.src = premotor[0]
p.dst = premotor[0]
p.connect()


### Feedback Premotor -> V4
print "Connecting premotor to V4"

for name in ['blue','green','PI_4','3PI_4']:
	p1.src1= premotor[0]
	exec("p1.src2 = I_"+name+"[0]")
	exec("p1.dst = V4_"+name+"[0]")
	p1.connect_point_mod_one(0.5)


#########################
## J. Vitay Connections
#########################

# Visual connections
####################
print "Binding visual connections"

p.profile = profile.gaussian(0.25,2.0/width)
p.shape = shape.disc(2.5*2.0/width)
p.src = percept[0]
p.dst = wm[0]
p.connect()

# CNFT connections
####################
print "Binding cnft connections"

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
p.src = premotor[0]
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
p.dst = premotor[0]
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
p.profile = profile.gaussian(-1.5,2.0/width)
p.shape = shape.disc(0.5)
p.src = gpi_inhib[0]
p.dst = thal_inhib[0]
p.connect()

# reward_inhib to striatum_inhib
p.profile = profile.constant(8.0)
p.shape = shape.box(1,1)
p.density = density.full(1.0)
p.src = switch[0]
p.dst = striatum_inhib[0]
p.connect()

# Anticpation connections
#########################
print "Binding anticipation connections"

# wm -- focus -> anticipation
p1.combination = combine.linear(1,-1,1,1,-1,1,-width/2.0,-height/2.0,0.05)
p1.src1= wm[0]
p1.src2 = premotor[0]
p1.dst = anticipation[0]
p1.connect();


## panticipation -> wm

p1.src1 = anticipation[0]
p1.src2 = percept[0]
p1.dst = panticipation[0]
p1.connect_point_mod_one(1.0)

p.profile = profile.gaussian(0.4,2.0/width)
p.shape = shape.disc(2.5*2.0/width)
p.src = panticipation[0]
p.dst = wm[0]
p.connect()



### anticipation -> wm
#p.profile = profile.gaussian(0.4,2.0/width)
#p.shape = shape.disc(2.5*2.0/width)
#p.src = anticipation[0]
#p.dst = wm[0]
#p.connect()

## anticipation <-> anticipation
p.profile = profile.dog(1.6,3.0/width,1.0,4.0/width)
p.shape = shape.disc(0.5)
p.src = anticipation[0]
p.dst = anticipation[0]
p.connect()


##### Unit specs
V4_blue.spec.alpha = 1.0
V4_green.spec.alpha = 1.0
V4_PI_4.spec.alpha = 1.0
V4_3PI_4.spec.alpha = 1.0
percept.spec.alpha = 0.6
percept.spec.tau = 2.0
percept.spec.baseline = -0.1
premotor.spec.alpha = 10.0
premotor.spec.baseline = 0.0

ITt_blue.spec.alpha = 1.0
ITt_green.spec.alpha = 1.0
ITt_PI_4.spec.alpha = 1.0
ITt_3PI_4.spec.alpha = 1.0

IT_blue.spec.alpha = 2.5
IT_green.spec.alpha = 2.5
IT_PI_4.spec.alpha = 2.5
IT_3PI_4.spec.alpha = 2.5

striatum_inhib.spec.baseline = -0.5
gpi_inhib.spec.baseline = 1.0#0.8

wm.spec.tau = 0.6
wm.spec.baseline = -0.25

#inhibition.spec.alpha = 13.2
inhibition.spec.baseline = -0.1
switch.spec.tau = 25
thal_wm.spec.tau = 0.6

anticipation.spec.alpha = 0.4
anticipation.spec.tau = 15.0

panticipation.spec.alpha = 1.0
panticipation.spec.tau = 0.75

# Tools

radius = 4

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

def clamp(map,x0,y0):
	global radius
	for u in map[0]:
		u.potential = u.potential +math.exp(-float((u.position[0]-x0)*(u.position[0]-x0)+(u.position[1]-y0)*(u.position[1]-y0))/float(radius*radius))
	return True

labels = ['green','blue','PI_4','3PI_4']
world = [[],[],[],[]]

def add_blob(feature,x,y):
	index = labels.index(feature);
	world[index].append([x,y])

def draw_world():
	clear_all()
	for i in range(len(world)):
		for j in range(len(world[i])):
			exec("clamp(I_"+labels[i]+","+str(world[i][j][0])+","+str(world[i][j][1])+")")

def clear(map):
	for u in map[0]:
		u.potential = 0
		
def clear_all():
	for name in ['I_blue','I_green','I_PI_4','I_3PI_4']:
		exec("clear("+name+")")
		
def switch_func():
	switch.unit(0).potential = 1.0

def move():
	global world,pan2do,tilt2do
	decode(premotor)
	for i in range(len(world)):
		for j in range(len(world[i])):
			world[i][j][0] = world[i][j][0] - pan2do
			world[i][j][1] = world[i][j][1] - tilt2do

def refresh():
	draw_world()

### Interface

def net_init(widget,data=None):
	add_blob('green',10,20)
	add_blob('blue',15,10)
	draw_world()

def net_evaluate(widget,data=None):
	net.evaluate(3,False)

def net_move(widget,data=None):
	move()

def net_refresh(widget,data=None):
	refresh()
	
def net_clear(widget,data=None):
	clear_all()	
	
def net_switch(widget,data=None):
	switch_func()


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
