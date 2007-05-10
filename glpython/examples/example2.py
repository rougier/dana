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

win = window(locals())

# Resize default view
win.view.set_size (.5,.7)
win.view.append (Cube())
win.view[0].use_border = True

# Add a new view on upper right
auxview = Viewport()
auxview.set_size (.5,.7)
auxview.set_position (-1,-1)
auxview.append (Background())
auxview[0].use_border = True
auxview.append (Cube())
win.viewport.append (auxview)

win.show()


