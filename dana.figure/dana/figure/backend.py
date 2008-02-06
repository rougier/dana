#!/usr/bin/env python
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
""" Generic figure backend

    A backend may be used to visualize a figure on screen and interact using
    mouse (zoom & drag). It must be specified for a specific GUI environment
    such as WX or GTK.

"""
import wx
from wx import glcanvas
import OpenGL.GL as GL
import OpenGL.GLU as GLU

class Backend:
    """ Generic figure backend """

    def __init__ (self, figure, fps=30.0):
        """ Create a new backend for the figure """

        self.figure = figure
        self.fps = fps
        if fps:
            self._delay = int(1000.0/float(self.fps))
        else:
            self._delay = 0
        self.size = 0,0
        self._initialized = False
        self._drag_in_process = False
        self._pointer = None
        self._select_callback = None

    def initialize (self):
        """ Initialize GL """
        self._initialized = True
        GL.glClearColor (1.0, 1.0, 1.0, 1.0)
        GL.glClearDepth (1.0)

    def setup (self):
        """ Setup projection and modelview """

        viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
        mode = GL.glGetIntegerv (GL.GL_RENDER_MODE)
        w,h = self.size

        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glLoadIdentity ()
        if (mode == GL.GL_SELECT) and (self.pointer_):
            GLU.gluPickMatrix (self.pointer_[0], self.pointer_[1], 1, 1, viewport);
        GL.glOrtho (0.0, w, 0.0, h, -1000.0, 1000.0)
        GL.glMatrixMode (GL.GL_MODELVIEW)
        GL.glLoadIdentity ()

    def render (self):
        """ Render figure """

        w,h = self.size
        size = max (w, h)/1.0
        size *= self.figure.zoom
        width  = w/size
        height = h/size
        dx = self.figure.position[0]
        dy = self.figure.position[1]

        GL.glClear (GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.setup()
        GL.glPushMatrix ()
        GL.glScalef (size, size, 1)
        GL.glTranslatef ((width -self.figure.normalized_size[0])/2.0 + dx,
                         (height-self.figure.normalized_size[1])/2.0 - dy,
                         0.0)
        self.figure.render ('opengl')
        GL.glPopMatrix ()

    def show (self):
        """ Show backend on screen and start event loop """
        pass

    def hide (self):
        """ Hide backend """
        pass

    def zoom_in (self):
        """ Zoom in figure """

        self.figure.zoom = self.figure.zoom * 1.1

    def zoom_out (self):
        """ Zoom out figure """

        self.figure.zoom = self.figure.zoom / 1.1


    def select (self, x, y):
        """ """
        
        GL.glSelectBuffer (512)
        GL.glRenderMode (GL.GL_SELECT)
        GL.glInitNames ()
        GL.glPushName (0)
        viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
        self.pointer_ = x, viewport[3]-y
        self.render()
        records = GL.glRenderMode (GL.GL_RENDER)
        if records:
            for record in records:
                if record[2]:
                    id = record[2][0]
                    if self._select_callback:
                        self._select_callback (id)
                        return
        self._select_callback (-1)
        self.pointer_ = None


    def drag_start (self, start):
        """ Start dragging figure from start point """

        self._drag_start = start
        self._drag_in_process = True
        self._initial_position = self.figure.position

    def drag_to (self, target):
        """ Drag figure to target point """

        if self._drag_in_process:
            dx = target[0] - self._drag_start[0]
            dy = target[1] - self._drag_start[1]
            self.drag = dx/self.figure.zoom, dy/self.figure.zoom
            self.figure.position = (self._initial_position[0] + self.drag[0],
                                    self._initial_position[1] + self.drag[1])
        else:
            self.drag = 0,0

    def drag_end (self):
        """ End dragging figure """

        if self._drag_in_process:
            self._drag_in_process = False
            self.drag = 0,0

