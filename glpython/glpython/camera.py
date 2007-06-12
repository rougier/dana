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
""" Camera abstraction

A camera describes the view matrix that is used for rendering a scene

Attributes:
    ortho: whether to use an ortho camera or not
    zoom: zoom factor
    near,far: near and far planes
    aperture: objective aperture
"""

import OpenGL.GL as GL
import OpenGL.GLU as GLU


class Camera:
    """ Camera abstraction """

    def __init__ (self):
        """ Initialize camera with default values """
        
        self.near = 1.0
        self.far = 100.0
        self.aperture = 30.0
        self.width, self.height = 0,0
        self.ortho = False
        self.zoom = 0.75

    def resize (self, width=None, height=None):
        """ Resize event """
        
        if width:
    	    self.width = width
        if height:
    	    self.height = height
    
        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glLoadIdentity ()

        mode =  GL.glGetIntegerv (GL.GL_RENDER_MODE)
        if mode == GL.GL_SELECT:
            viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
            GLU.gluPickMatrix (self.x, self.y, 1, 1, viewport)

        if self.ortho:
            if self.height == 0:
                aspect = self.width
            else:
                aspect = self.width/float(self.height)
            if aspect > 1.0:
                left = -aspect
                right = aspect
                bottom = -1
                top =  1
            else:
                left = -1.0
                right =  1.0
                bottom = -1.0/aspect
                top =  1.0/aspect
            GL.glOrtho (left, right, bottom, top, self.near, self.far)
        else:
            if height == 0:
                aspect = self.width
            else:
                aspect = self.width/float(self.height)
            GLU.gluPerspective (self.aperture, aspect, self.near, self.far)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity ()
        GL.glTranslatef (0,0,-5)
        if self.zoom < 0.1:
            self.zoom = 0.1
        elif self.zoom > 5:
            self.zoom = 5
        GL.glScalef (self.zoom*2, self.zoom*2, self.zoom*2)

