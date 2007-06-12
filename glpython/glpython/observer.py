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
""" Observer abstraction 

An observer is a point in space describing the position of a virtual observer
looking at point (0,0,0) using a specific camera and can be moved using mouse.

Attributes:
    phi:   rotation around x axis
    theta: rotation aoound z axis (0 to pi)
"""

import OpenGL.GL as GL
import camera


class Observer:
    """ Observer abstraction """
    
    def __init__ (self):
        """ Intialize observer with default values """
        
        self.phi = -45.0
        self.theta = 65
        self.ox, self.oy = 0, 0
        self.camera = camera.Camera()
        self.button = 0

    def look (self):
        """ Make observer to actually look """

        GL.glRotatef (-self.theta, 1, 0, 0)
        GL.glRotatef (self.phi, 0, 0, 1)

    def resize (self, width=None, height=None):
        """ Resize event """

        self.camera.resize (width, height)

    def move_start (self, x, y, button):
        """ Mouse move start """

        self.ox, self.oy, self.button = x, y, button

        
    def move_end (self, x, y):
        """ Mouse move end """

        self.ox, self.oy, self.button = -1, -1, 0

    def move_to (self, x, y):
        """ Mouse move """
        
        # Move
        if self.button == 1:
            if self.ox < 0:
                return
            self.phi += (x - self.ox) / 4.0
            self.theta += (y - self.oy) / 4.0     
            if self.theta > 180.0:
                self.theta = 180.0
            elif self.theta < 0.0:
                self.theta = 0.0
        # Zoom
        elif self.button == 2:   
            self.camera.zoom += ((y-self.oy)/float(self.camera.height))*3
        self.ox, self.oy = x, y


