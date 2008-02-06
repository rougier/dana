#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# DANA 
# Copyright (C) 2006-2007  Nicolas P. Rougier
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
#------------------------------------------------------------------------------

import os.path
import OpenGL.GL as GL
import font
import _gl
from .. import data


class Array (_gl.Array):

    def __init__(self, array):
        _gl.Array.__init__ (self, array.array, array.cmap, array.position, array.size)
        self.array = array
        self.font = None


    def render (self, userdata=None):
        _gl.Array.render(self)

        if not self.font:
            fontfile = os.path.join (os.path.dirname (data.__file__), "sans.ttf")
            self.font = font.Font (fontfile)

        if not self.array.name:
            return

        viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
        w = float (viewport[2] - viewport[0])
        h = float (viewport[3] - viewport[1])

        size = self.array.fontsize * 0.00666
        GL.glPushMatrix ()
        GL.glTranslatef (self.array.position[0] + 0.05*self.array.size[0],
                         self.array.position[1] + 0.05*self.array.size[1],
                         0)
        GL.glScalef (size, size, 1)
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable (GL.GL_TEXTURE_2D)
        GL.glEnable (GL.GL_BLEND)
	GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glTexEnvf (GL.GL_TEXTURE_ENV,
                      GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
        GL.glColor (0,0,0)
        self.font.render (self.array.name)
        GL.glPopMatrix()
        GL.glDisable (GL.GL_TEXTURE_2D)
        GL.glDisable (GL.GL_BLEND)

