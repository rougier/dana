#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# DANA -- Distributed Asynchronous Numerical Adaptive computing library
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
""" OpenGL reference frame object """

import os.path
import OpenGL.GL as GL
import font
from .. import data


class Frame:
    """ Figure frame of reference """

    def __init__(self, frame):
        """ Create a new frame """        

        self.frame = frame


    def render (self, userdata=None):
        """ Render the frame """

        GL.glDisable (GL.GL_LINE_SMOOTH)
	GL.glDisable (GL.GL_BLEND)
        GL.glDisable (GL.GL_TEXTURE_2D)
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glColor (0,0,0)
        GL.glBegin (GL.GL_QUADS)
        GL.glVertex (0,                        0,                        0)
        GL.glVertex (0.999*self.frame.size[0], 0,                        0)
        GL.glVertex (0.999*self.frame.size[0], 0.999*self.frame.size[1], 0)
        GL.glVertex (0,                        0.999*self.frame.size[1], 0)
        GL.glEnd()
