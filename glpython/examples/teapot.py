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

from glpython import *
import sys
import OpenGL.GL as GL
import OpenGL.GLUT as GLUT

class Teapot (Object):
    def __init__(self):
        Object.__init__(self)
        self.color = (.75,.75,.75)
        
    def render (self):
        GL.glPushMatrix()
        GL.glRotatef (90, 1,0,0)

        GL.glColor ( self.color )
        GL.glPolygonOffset (1,1)
        GL.glEnable (GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GLUT.glutSolidTeapot (.5)

        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glDisable (GL.GL_POLYGON_OFFSET_FILL)
        GL.glColor ( (.25,.25,.25) )
        GLUT.glutSolidTeapot (.5)
        GL.glPopMatrix()


teapot = Teapot()
cmap = Colormap()
cmap.append (1, (0,0,1,1))
cmap.append (0, (1,1,1,1))
bg = Background (cmap=cmap, alpha=.1)
win, fig = window (size = (800,600), title='teapot.py',
                   has_terminal=True, namespace = locals())
fig.append (bg)
fig.append (teapot)
GLUT.glutInit(sys.argv)
win.show()


