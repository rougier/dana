#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# DANA -- Distributed Asynchronous Numerical Adaptive computing library
# Copyright (c) 2007  Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

import os.path
import OpenGL.GL as GL
import font
from .. import data

class Colorbar:

    def __init__(self, colorbar):
        """ """

        self.font = None
        self.colorbar = colorbar
        self.list = None


    def render (self, userdata=None):
        """ """

        if self.list:
            GL.glCallList (self.list)

        if not self.font:
            fontfile = os.path.join (os.path.dirname (data.__file__), "sans.ttf")
            self.font = font.Font (fontfile)

        self.list = GL.glGenLists (1)
        GL.glNewList (self.list, GL.GL_COMPILE)

        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glDisable (GL.GL_TEXTURE_2D)

        size = self.colorbar.size
        position = self.colorbar.position
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # Background
        w,h = self.colorbar.size
        x,y = self.colorbar.position
        cmap = self.colorbar.cmap

        for i in xrange (25):
            GL.glBegin (GL.GL_QUADS)        
            x1 = x
            x2 = x + w
            y1 = y + h*i/25.0
            y2 = y + h*(i+1)/25.0
                
            c = cmap.color(cmap.min + i/25.0 * (cmap.max-cmap.min))
            r,g,b = c.red, c.green, c.blue
            GL.glColor (r,g,b)
            GL.glVertex (x1, y1, -75)
            GL.glVertex (x2, y1, -75)

            c = cmap.color(cmap.min + (i+1)/25.0 * (cmap.max-cmap.min))
            r,g,b = c.red, c.green, c.blue
            GL.glColor (r,g,b)
            GL.glVertex (x2, y2, -75)
            GL.glVertex (x1, y2, -75)
            GL.glEnd()
            
        # Outer frame
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glColor (0,0,0)
        GL.glBegin (GL.GL_QUADS)
        GL.glVertex (position[0],         position[1],         -50)
        GL.glVertex (position[0]+size[0], position[1],         -50)
        GL.glVertex (position[0]+size[0], position[1]+size[1], -50)
        GL.glVertex (position[0],         position[1]+size[1], -50)
        GL.glEnd()
            
        # Ticks
        s = size[1]/self.colorbar.tick_number
        for i in xrange(int(self.colorbar.tick_number)):
            GL.glBegin (GL.GL_LINES)
            GL.glVertex (position[0],             position[1]+(i+1)*s, -25)
            GL.glVertex (position[0]+.15*size[0], position[1]+(i+1)*s, -25)
            GL.glEnd()

            GL.glBegin (GL.GL_LINES)
            GL.glVertex (position[0]+size[0],              position[1]+(i+1)*s, -25)
            GL.glVertex (position[0]+size[0]-0.15*size[0], position[1]+(i+1)*s, -25)
            GL.glEnd()

        # Labels
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable (GL.GL_TEXTURE_2D)
        GL.glEnable (GL.GL_BLEND)
	GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glTexEnvf (GL.GL_TEXTURE_ENV,
                      GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)

        viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
        w = float (viewport[2] - viewport[0])
        h = float (viewport[3] - viewport[1])
        size = max(self.colorbar.size[0], self.colorbar.size[1])/150.0

        # Y axis labels
        for i in xrange(int(self.colorbar.tick_number+1)):
            GL.glPushMatrix ()
            GL.glTranslatef (self.colorbar.position[0]+self.colorbar.size[0], i*s, 0)
            GL.glScalef (1.0/self.colorbar.tick_number*size,
                         1.0/self.colorbar.tick_number*size,
                         1)
            v = cmap.min + i/float(self.colorbar.tick_number) * (cmap.max-cmap.min)
            if v >=0:
                tw,th = self.font.extents  (" +%.2f" % v)
                GL.glTranslatef (0, -th/2, 0)
                self.font.render (" +%.2f" % v)
            else:
                tw,th = self.font.extents  (" %.2f" % v)
                GL.glTranslatef (0, -th/2, 0)
                self.font.render (" %.2f" % v)
            GL.glPopMatrix ()
        GL.glDisable (GL.GL_TEXTURE_2D)
        GL.glDisable (GL.GL_BLEND)
        GL.glEndList()
