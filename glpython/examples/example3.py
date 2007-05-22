#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
#------------------------------------------------------------------------------


from glpython.window import window
from glpython.viewport import Viewport
from glpython.cube import Cube
from glpython.background import Background

win = window(locals(), fps=30.0)

# Resize default view
win.view.set_size (1.0,.7)
win.view.append (Cube())
win.view[0].use_border = False

# Add a new view to default view
auxview = Viewport()
auxview.set_size (.5,.5)
auxview.set_position (-1,-1)
background = Background()
background.bg_color_top = (1,1,1,.75)
background.bg_color_bottom = (1,1,1,.75)
auxview.append (background)
auxview[0].use_border = True
auxview.append (Cube())
win.view.append (auxview)

win.show()


