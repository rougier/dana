#!/usr/bin/env python

#import numpy
from glpython import *
from glpython.world.core import *
from glpython.world.objects import *

roger = Robot()
win,fig = window (size=(800,600), title = "World",has_terminal=True,namespace=locals())

# Set the view of the robot
roger.view.position = (-1,0.5)
roger.view.size = (.25,.25)

# Append the 3rd person view
# and the view of the robot
fig.append(roger.view)

# Append a yellow disc to the scene
# pointing toward the robot
d = Disc()
d.phi = 90
d.radius = 0.4
d.z = 1.0
d.color[0] = 1
d.color[1] = 1
d.color[2] = 0
roger.view.append(d)

# Append bars to the scene
# A 90 degrees, blue bar, at (0,0)
bb = Bar()
bb.color[0] = 0
bb.color[1] = 0
bb.color[2] = 1
bb.theta = 90
bb.y = 0
bb.z = 0
roger.view.append(bb)

# A 135 degrees, red bar, at (-1,0)
rb = Bar()
rb.color[0] = 1
rb.color[1] = 0
rb.color[2] = 0
rb.theta = 135
rb.y = -1
rb.z = 0
roger.view.append(rb)


win.show()
