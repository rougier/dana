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

# Maps Hamker
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

#pf = new_map('pf', 1,1,-2,-1,neural.Unit)
#pf.unit(0).value = 1.0

#V4_to_IT = new_map('V4_to_IT', 1, 1, -3, -1, neural.Unit)
#V4_to_IT.unit(0).value = 1.0

V4_to_percept = new_map('V4_to_percept', 1, 1, 1, 0, core.Unit)
V4_to_percept.unit(0).potential = 1.0

#PF_to_trouve = new_map('PF_to_trouve', 1, 1, -5, -1, neural.Unit)
#PF_to_trouve.unit(0).value = 0.0

percept = new_map('Perceptual', width, height, 5, 2, sigmapi.Unit)
premotor = new_map('Premotor', width, height, 6, 2, cnft.Unit)

print "Map built"

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


### inhibition
#print "V4 lateral inhibition, modulated by ITt"

##l1.set_source(V4_blue)
##l1.set_destination(V4_green)
##l1.set_modulator(ITt_blue)
##l1.connect_as_modulator_gaussian(4,-0.2)
##l1.connect_as_modulatorSingle(-0.1,1)

##l1.set_source(V4_green)
##l1.set_destination(V4_blue)
##l1.set_modulator(ITt_green)
##l1.connect_as_modulatorSingle(-0.1,1)
##l1.connect_as_modulator_gaussian(4,-0.2)

##l1.set_source(V4_3PI_4)
##l1.set_destination(V4_PI_4)
##l1.set_modulator(ITt_3PI_4)
##l1.connect_as_modulatorSingle(-0.1,1)
##l1.connect_as_modulator_gaussian(4,-0.2)

##l1.set_source(V4_PI_4)
##l1.set_destination(V4_3PI_4)
##l1.set_modulator(ITt_PI_4)
##l1.connect_as_modulatorSingle(-0.1,1)
##l1.connect_as_modulator_gaussian(4,-0.2)

#####################
######## What Pathway

#print "What pathway : V4 to IT"
### Feedforward : V4 -> IT
print "Connecting V4 to IT"

for name in ['blue','green','PI_4','3PI_4']:
	exec("p1.src1 = V4_"+name+"[0]")
	exec("p1.dst = IT_"+name+"[0]")
	p1.connect_all_to_one(1.0)


####  Spatial attention biaises the activity in IT
#l1.set_source(V4_blue)
#l1.set_destination(IT_blue)
#l1.set_modulator(premotor)
#l1.connect_as_sigmaPi_to_one(1.0,3)

#l1.set_source(V4_green)
#l1.set_destination(IT_green)
#l1.set_modulator(premotor)
#l1.connect_as_sigmaPi_to_one(1.0,3)

#l1.set_source(V4_PI_4)
#l1.set_destination(IT_PI_4)
#l1.set_modulator(premotor)
#l1.connect_as_sigmaPi_to_one(1.0,3)

#l1.set_source(V4_3PI_4)
#l1.set_destination(IT_3PI_4)
#l1.set_modulator(premotor)
#l1.connect_as_sigmaPi_to_one(1.0,3)


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


##### Unit specs
V4_blue.spec.alpha = 1.0
V4_green.spec.alpha = 1.0
V4_PI_4.spec.alpha = 1.0
V4_3PI_4.spec.alpha = 1.0
percept.spec.alpha = 1.0
percept.spec.tau = 2.0
percept.spec.baseline = -0.1
premotor.spec.alpha = 4.7
premotor.spec.baseline = 0.0

ITt_blue.spec.alpha = 1.0
ITt_green.spec.alpha = 1.0
ITt_PI_4.spec.alpha = 1.0
ITt_3PI_4.spec.alpha = 1.0

IT_blue.spec.alpha = 2.5
IT_green.spec.alpha = 2.5
IT_PI_4.spec.alpha = 2.5
IT_3PI_4.spec.alpha = 2.5
##############
#### Attention
##############

##print "Binding visual connections"

## percept to fef
#l.set_source (percept)
#l.set_destination (fef)
#l.connect_as_gaussian (2, 0.5);

##print "Binding cnft connections"
## fef
#l.set_source (fef)
#l.set_destination (fef)
#l.connect_as_dog (2, 2.5, 4, 1);

##print "Binding inter-cortical connections"
## focus to fef
#l.set_source (premotor)
#l.set_destination (fef)
#l.connect_as_gaussian (2, 0.2);

##print "Binding working memory connections"
## fef to thal_fef
#l.set_source (fef)
#l.set_destination (thal_fef)
#l.connect_as_gaussian (1.5, 2.35);

## thal_fef to fef
#l.set_source (thal_fef)
#l.set_destination (fef)
#l.connect_as_gaussian (1.5, 2.4);

##print "Binding anticipation connections"

####### CONNECTIONS ANTICIPATRICES DESACTIVEES

## fef -> anticipated_fef
##l1.set_source(fef)
##l1.set_destination(anticipated_fef)
##l1.set_modulator(consigne)
##l1.set_offset(width/2,height/2)
##l1.connect_as_modulator_motor(0.05,1,1,1,0)

## anticipated_fef -> fef
##l.set_source(anticipated_fef)
##l.set_destination(fef)
##l.connect_as_gaussian(2,0.2)

## anticipated_fef <-> anticipated_fef
##l.set_source(anticipated_fef)
##l.set_destination(anticipated_fef)
##l.connect_as_dog(3.0,1.6,4.0,1)

## consigne to has_consigne
#l.set_source(consigne)
#l.set_destination(has_consigne)
#l.connect_as_square(0.3,width)

## focus to consigne
#l1.set_source(premotor)
#l1.set_destination(consigne)
#l1.connect_as_central_sym(1.0)

## motor_gate to has_consigne
#l.set_source(motor_gate)
#l.set_destination(has_consigne)
#l.connect_as_point(-5)

##### switch mechanism
#l.set_source(inhib)
#l.set_destination(premotor)
#l.connect_as_gaussian(2.0,-1.0)

#l1.set_source(fef)
#l1.set_destination(inhib)
#l1.set_modulator(switch)
#l1.connect_as_modulatorSingle(1.0,1)

##### Unit specs

#consigne.spec.tau = 5.0
#consigne.spec.alpha = 1.0

#motor_gate.spec.baseline = 0.8
#motor_gate.spec.tau = 2.0

#ROGER_CENTER_X = 0
#ROGER_CENTER_Y = 0

#ROGER_X_FACTOR = 1.0
#ROGER_Y_FACTOR = 1.0

#CONSIGNE_THRES = 0.4

#print "Links OK"

#########################
## Simulation Parameters
#########################
#pan2do = 0.0
#tilt2do = 0.0
#is_connected = 0
#CLEAR_REFRESH = 2000

#PF_COLOR_VALUE = 1.0
#PF_ORIENTATION_VALUE = 1.0
####
## Robot
#######
#Roger = roger.Robot(envs)
#net.add(Roger)
#Roger.attach(V4_blue)

## OpenGL View
#view = gui.View (net)
#view.show()

## Simulation control panel
#net.gui = gui.NetDialog (net)
#net.gui.show()

#def net_start(widget,data=None):
    #net.start(0)
    #consigne_spy()

#def motor_step():
    #motor_gate.unit(0).value = 0

#def consigne_spy():
    #if(has_consigne.unit(0).value >= CONSIGNE_THRES):
        #print "mouvement declenche !"
        #decode_consigne()
        #move_camera()
    #gtk.timeout_add(250,consigne_spy)
    
#def connect_roger(widget,data=None):
    #print "Connecting to roger ..."
    #if(Roger.connect("coupelle.loria.fr",8000)):
        #print "Connected !"
        #global is_connected
        #is_connected = 1
        #gtk.timeout_add(1000,center)
        #Roger.addFilter("HSV_Blue",I_blue)
        #Roger.addFilter("HSV_Green",I_green)
        #Roger.addFilter("SobelPI_4",I_PI_4)
        #Roger.addFilter("Sobel3PI_4",I_3PI_4)
    #else:
        #print "Connecting problem !"

#def disconnect_roger(widget,data=None):
    #print "Disconnecting from roger .."
    #Roger.disconnect()
    #global is_connected
    #is_connected = 0

#def refresh():
    #if(is_connected):
        #print "On rafraichit la video"
        #clear_inputs()
        #gtk.timeout_add(CLEAR_REFRESH,Roger.grabImage,"")
    #else:
        #print "Please connect me first to roger!"

#def clear_inputs():
    #I_blue.clear()
    #I_green.clear()
    #I_PI_4.clear()
    #I_3PI_4.clear()

#def center():
    #Roger.moveCamera(ROGER_CENTER_X,ROGER_CENTER_Y)

#def decode_consigne():
    #global pan2do
    #global tilt2do
    #pan2do = 0.0
    #tilt2do = 0.0
    #activite_totale = 0.0
    #for i in xrange(width):
        #for j in xrange(height):
            #pan2do += consigne.unit(i,j).value*(i-width/2)
            #tilt2do += consigne.unit(i,j).value*(j-height/2)
            #activite_totale += consigne.unit(i,j).value
    #if(activite_totale!=0.0):
        #pan2do /= -activite_totale ### !!! Signe - pour qu'il fasse le mouvement dans le bon sens
        #tilt2do /= -activite_totale
    #else:
        #pan2do = 0.0
        #tilt2do = 0.0
        #print "[Warning] : decodage de la carte focus; activite totale nulle !!"

#def move_camera():
    #print "On fait le mouvement de camera"
    #global pan2do,tilt2do
    #Roger.moveCameraDelta(pan2do*ROGER_X_FACTOR,tilt2do*ROGER_Y_FACTOR)    

#def enable_blue(widget,data=None):
    #PF_blue.unit(0).value = PF_COLOR_VALUE

#def disable_blue(widget,data=None):
    #PF_blue.unit(0).value = 0.0

#def enable_green(widget,data=None):
    #PF_green.unit(0).value = PF_COLOR_VALUE

#def disable_green(widget,data=None):
    #PF_green.unit(0).value = 0.0

#def enable_PI_4(widget,data=None):
    #PF_PI_4.unit(0).value = PF_ORIENTATION_VALUE

#def disable_PI_4(widget,data=None):
    #PF_PI_4.unit(0).value = 0.0

#def enable_3PI_4(widget,data=None):
    #PF_3PI_4.unit(0).value = PF_ORIENTATION_VALUE

#def disable_3PI_4(widget,data=None):
    #PF_3PI_4.unit(0).value = 0.0

#def clear_premotor(widget,data=None):
    #premotor.clear()

#def switch_target (widget, data=None):
    #switch.unit(0).value = 0.75

#def mem_and_search(widget, data=None):
    #PF_blue.unit(0).value = IT_blue.unit(0).value
    #PF_green.unit(0).value = IT_green.unit(0).value
    #PF_PI_4.unit(0).value = IT_PI_4.unit(0).value
    #PF_3PI_4.unit(0).value = IT_3PI_4.unit(0).value
    #clear_inputs()

#TIMEOUT_COMPARE = 3000
#THRESHOLD_COMPARE = 0.65

#def demo():
    #print "On recherche le stimulus ayant les features :"
    #print " Bleu :",PF_blue.unit(0).value
    #print " Vert :",PF_green.unit(0).value
    #print "  /   :",PF_PI_4.unit(0).value
    #print "  \   :",PF_3PI_4.unit(0).value
    #demo_compare()
    #print "demo terminée"
    
#def demo_compare():
    #print " je compare "
    #if(trouve.unit(0).value >= THRESHOLD_COMPARE):
        #decode_consigne()
        #move_camera()
    #else:
        #switch.unit(0).value = 0.75
        #gtk.timeout_add(TIMEOUT_COMPARE,demo_compare)
    
#window = gtk.Window(gtk.WINDOW_TOPLEVEL)
#window.set_border_width(12)
#vbox = gtk.VBox(True, 6)
#table = gtk.Table()
#label = gtk.Label("Prefontral")

#connect_button = gtk.Button("Connect to Roger")
#connect_button.connect("clicked", connect_roger)
#table.attach(connect_button , 0 , 1 , 0 , 1)

#disconnect_button = gtk.Button("Disconnect to Roger")
#disconnect_button.connect("clicked", disconnect_roger)
#table.attach(disconnect_button , 1 , 2 , 0 , 1)

#table.attach(label, 0 , 2, 1 , 2,gtk.EXPAND|gtk.FILL, gtk.EXPAND|gtk.FILL, 0, 6)

#blue_button = gtk.Button("Blue")
#blue_button.connect("clicked", enable_blue)
#table.attach(blue_button , 0 , 1 , 2 , 3)

#no_blue_button = gtk.Button("No Blue")
#no_blue_button.connect("clicked", disable_blue)
#table.attach(no_blue_button , 1 , 2 , 2 , 3)

#green_button = gtk.Button("Green")
#green_button.connect("clicked", enable_green)
#table.attach(green_button , 0 , 1 , 3 , 4)

#no_green_button = gtk.Button("No Green")
#no_green_button.connect("clicked", disable_green)
#table.attach(no_green_button , 1 , 2 , 3 , 4)

#PI_4_button = gtk.Button("Pi/4")
#PI_4_button.connect("clicked", enable_PI_4)
#table.attach(PI_4_button , 0 , 1 , 4 , 5)

#no_PI_4_button = gtk.Button("No Pi/4")
#no_PI_4_button.connect("clicked", disable_PI_4)
#table.attach(no_PI_4_button , 1 , 2 , 4 , 5)

#TPI_4_button = gtk.Button("3Pi/4")
#TPI_4_button.connect("clicked", enable_3PI_4)
#table.attach(TPI_4_button , 0 , 1 , 5 , 6)

#no_TPI_4_button = gtk.Button("No 3Pi/4")
#no_TPI_4_button.connect("clicked", disable_3PI_4)
#table.attach(no_TPI_4_button , 1 , 2 , 5 , 6)

#premotor_button = gtk.Button("Clear Premotor")
#premotor_button.connect("clicked", clear_premotor)
#table.attach(premotor_button, 0, 2, 6, 7)

#switch_button = gtk.Button("Switch")
#switch_button.connect("clicked", switch_target)
#table.attach(switch_button, 0, 2, 7, 8)

#mem_button = gtk.Button("Mem & Search")
#mem_button.connect("clicked", mem_and_search)
#table.attach(mem_button, 0, 2, 8, 9)

#start_button = gtk.Button("Start")
#start_button.connect("clicked", net_start)
#table.attach(start_button, 0, 2, 9, 10)

#window.add (table)
#window.show_all()

# Tools

radius = 4

def clamp(map,x0,y0):
	global radius
	for u in map[0]:
		u.potential = u.potential +math.exp(-float((u.position[0]-x0)*(u.position[0]-x0)+(u.position[1]-y0)*(u.position[1]-y0))/float(radius*radius))
	return True

def clear(map):
	for u in map[0]:
		u.potential = 0
		
def clear_all():
	print "clear all"	
		
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

# Show network
netview = view.view (net)

manager = pylab.get_current_fig_manager()

def updatefig(*args):
    netview.update()
    return True

gobject.idle_add(updatefig)
pylab.show()
