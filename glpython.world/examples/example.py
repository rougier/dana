#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier - Jeremy Fix.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
#------------------------------------------------------------------------------


# The top left figure (fig1) is a third person view. You can navigate in it
# by focusing it, and using the following keybindings :
# 'up' : move forward
# 'down' : move backward
# 'left' : straff left
# 'right' : straf right
# 'home' : move up
# 'end' : move down
# 'button click + mouse move' : rotate the camera
#
# The figure at the bottom (roger.view) of the previous one is a first personn view
# inside the robot.
# You can move the robot by using one of the following commands :
# roger.move(x,y,z) : move forward (backward) of x (-x) steps
#                     straff left (right) of y (-y) steps
#                     move up (down) of z (-z) steps
# roger.rotate(pan,tilt) : rotate the robot
# roger.rotateCamera(pan,tilt) : rotate the camera only
# roger.grab("filename.ppm") : grab the view of the camera and save it to filename.ppm

# Be carefull when you want to add an object to the scene,
# the object must be added to 2 viewports :
# myobject o
# fig1.append(o)
# roger.view.append(o)
# If not, the object will only appear in one of the viewports


from glpython import *
from glpython.world.core import *
from glpython.world.objects import *

roger = Robot()
win,fig = window (size=(800,600), title = "World",has_terminal=True,namespace=locals())


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
        
win.__class__.key_press = key_press_perso


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


win.show()
