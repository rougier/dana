#!/usr/bin/env python

import matplotlib.pylab as pylab
import matplotlib.colors as colors

import dana.core as core
import dana.projection as proj
import dana.cnft as cnft
import dana.view as view

import time, random, math
import gobject, gtk


print "--------------------------------------------------------------------"
print "CNFT using full connectivity"
print ""
print "Author:    Nicolas Rougier"
print "Date:      01/03/2005"
print "Reference: Rougier N.P. & Vitay J."
print "           'Emergence of Attention within a Neural Population'"
print "           Neural Networks, 19, 5, pp 573-581, June 2006."
print "--------------------------------------------------------------------"
print ""


# Create a new network
net = core.Network ()
width  = 40
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
Focus.spec.alpha    = 13.0
Focus.spec.min_act  = 0.0
Focus.spec.max_act  = 1.0

net.append(Focus)

# Create input to focus connections
p = proj.projection()
p.distance = proj.distance.euclidean (True)
p.density = proj.density.full(1)
p.profile = proj.profile.constant(1.0)
p.shape = proj.shape.point()
p.src = Input[0]
p.dst = Focus[0]
p.connect()

# Create focus laterals connections
p.self = False
#p.density = proj.density.sparser(.5)
p.profile = proj.profile.dog (2.20, 3.0/width, 0.65, 11.0/width)

p.shape = proj.shape.box(1,1)
p.src = Focus[0]
p.dst = Focus[0]
p.connect()

for u in Input[0]:
    u.potential = random.uniform(0.0, 1.0)

for i in xrange(Input.shape[0]):
    for j in xrange(Input.shape[1]):
        x0 = i/float(Input.shape[0])-.25
        y0 = j/float(Input.shape[1])-.25
        x1 = i/float(Input.shape[0])-.75
        y1 = j/float(Input.shape[1])-.75
        Input[0].unit(i,j).potential =  + math.exp (-(x0*x0+y0*y0)/0.0125) + math.exp (-(x1*x1+y1*y1)/0.0125) + .15*random.uniform(0.0, 1.0)
        
# Show network
netview = view.view (net)

manager = pylab.get_current_fig_manager()
cnt = 0
tstart = time.time()

def updatefig(*args):
#    global cnt, start,net
#    net.evaluate(1,False)
    netview.update()
#    cnt += 1
#    if cnt==500:
#        print 'FPS', cnt/(time.time() - tstart)
#        return False
    return True
cnt = 0

gobject.idle_add(updatefig)
pylab.show()


