#!/usr/bin/env python

#------------------------------------------------------------------------------
#
#   Copyright (c) 2007 Nicolas Rougier
# 
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
# 
#------------------------------------------------------------------------------
""" Background object

A background is a simple filled rectangular area (relative to window) area
with a border.
"""

import OpenGL.GL as GL
from object import Object


class Background (Object):
    """ Background object """
    
    def __init__(self):
        """ Intialization with default values"""
        
        Object.__init__(self)
        self.bg_color_bottom = (1, 1, 1, 0)
        self.bg_color_top = (1, 1, 1, 0)
        self.br_color = (0,0,0,1)
        self.use_border = False

    
    def render(self):
        """ Render the background with proper "2D" view """
        
        viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
        w,h = viewport[2], viewport[3]

        # Switch to '2D' mode
        GL.glPushAttrib (GL.GL_ENABLE_BIT)        
        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glPushMatrix ()
        GL.glLoadIdentity ()
        GL.glOrtho (0, w, 0, h, -100, 100)
        GL.glMatrixMode (GL.GL_MODELVIEW)
        GL.glPushMatrix ()
        GL.glLoadIdentity()

        # Disable what is not necessary
        GL.glDisable (GL.GL_LIGHTING)
        GL.glDisable (GL.GL_TEXTURE_2D)
        GL.glEnable (GL.GL_BLEND)
        GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable (GL.GL_DEPTH_TEST)

        # Background
        GL.glTranslatef (0.0, 0.0, -99)
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glBegin (GL.GL_QUADS)
        GL.glColor (self.bg_color_bottom)
        GL.glVertex2i (0, 0)
        GL.glVertex2i (w-1, 0)
        GL.glColor (self.bg_color_top)
        GL.glVertex2i (w-1, h-1)
        GL.glVertex2i (0, h-1)
        GL.glEnd()

        # Border
        if self.use_border:
            GL.glTranslatef (0.0, 0.0, 1)        
            GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glBegin (GL.GL_QUADS)
            GL.glColor (self.br_color)
            GL.glVertex2i (0, 0)
            GL.glVertex2i (w-1, 0)
            GL.glVertex2i (w-1, h-1)
            GL.glVertex2i (0, h-1)
            GL.glEnd()        
 
        # Go back to previous projection and view
        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glPopMatrix ()
        GL.glMatrixMode (GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glPopAttrib()





