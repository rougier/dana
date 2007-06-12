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

""" OpenGL Window

This module provides a single 'window' function that creates a graphical
window with an OpenGL context displaying a basic scene and handling most
events.
"""


import sys
from OpenGL.GL import *
from OpenGL.GL.EXT.framebuffer_object import *
from optparse import OptionParser
from glpython import backends
from glpython.viewport import Viewport
from glpython.terminal import Terminal
from glpython.background import Background

_window = None


def window (namespace={}, w=850, h=600, title='GLPython', fps=30,
            backend='wx', shell = 'ipython', layout = '1', logo=False):
    """ Return a window with OpenGL capabilities displaying a basic scene """

    global _window
    
    if _window:
        return _window

    base,backend = backends.WindowClass (backend)
    
    class Window (base):
        """ """
        def __init__ (self, w=w, h=h, title='OpenGL window', fps=30.0):
            """ """
            base.__init__ (self,w,h,title,fps)
            self.initialized = False
            self.outer_mainloop = False
            self.viewport = Viewport (name='main', reset=self.reset)
            self.viewport.bg_color = (1,1,1,0)
            self.viewport.use_border = False
            self.terminal = None

        def init (self):
            """ """
            self.viewport.init()
            if self.terminal:
                self.terminal.init()

        def reset (self):
            """ Reset view """
            if self.initialized:
                self.resize (self.width, self.height)

        def resize (self, w, h):
            """ """
            if not self.initialized:
                self.init()
                self.initialized = True
            self.width, self.height = w,h
            glViewport (0, 0, self.width, self.height)
            self.viewport.resize (w,h)
            if self.terminal:
                self.terminal.resize (self.width, self.height)

        def render (self):
            """ """
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glViewport (0, 0, self.width, self.height)
            if self.terminal:
                self.terminal.render()
            self.viewport.render ()

        def key_press (self, key):
            """ """
            if key == 'control-d':
                if not self.terminal or self.terminal.rl.line == '':
                    self.destroy()
            elif key == 'control-f' or key == 'f11':
                if self.fullscreened:
                    self.unfullscreen()
                    self.fullscreened = False
                else:
                    self.fullscreen()
                    self.fullscreened = True
            elif key == 'f1':
                self.set_layout ('1')
            elif key == 'f2':
                self.set_layout ('2')
            elif key in ['f10','escape']:
                self.viewport.key_press(key)
            elif key == 'f9':
                self.viewport.key_press(key)
            elif self.terminal:
                self.terminal.key_press (key)

        def button_press (self, button, x, y):
            """ """
            if button == 3:
                self.select (x,y)
            elif button == 4 and self.terminal:
                self.terminal.scroll_up()
            elif button == 5 and self.terminal:
                self.terminal.scroll_down()            
            else:
                self.viewport.button_press (button,x,y)

        def button_release (self, x, y):
            """ """
            self.viewport.button_release (x,y)

        def mouse_motion (self, x, y):
            """ """
            if self.terminal:
                self.terminal.check_focus (x,y)
            self.viewport.mouse_motion (x,y)
            if self.viewport.has_focus() and self.viewport.focus == False:
                if self.terminal:
                    self.terminal.focus = False
            

        def select (self, x, y):
            """ Select """
            glSelectBuffer (512)
            glRenderMode (GL_SELECT)
            glInitNames () 
            glPushName (0)

            glPushMatrix()
            glLoadIdentity()
            self.viewport.selection (x,y)
            self.viewport.render()
            buffer = glRenderMode (GL_RENDER)

            names = []
            for hit_record in buffer:
                min_depth, max_depth, name = hit_record
                if name[0] > 0:
                    names.append (name[0])
            names += [0,0]

            if names[0]:
                self.viewport.select (names[0], names[1])
            else:
                self.viewport.unselect ()

            glPopMatrix()
            glPopName()
            self.reset()    
    
        def set_layout (self, layout):
            """ Set layout """

            if layout == '1':
                self.view.set_size (1.0,1.0)
                self.view.set_position (0,0)
                if self.terminal:
                    self.terminal.set_size (1.0,1.0)
                    self.terminal.set_position (0,0)
                    self.terminal.use_border = False
            elif layout == '2':
                if self.terminal:
                    self.terminal.set_size (1.0,0.249)
                    self.terminal.set_position (0,0)
                    self.terminal.use_border = True
                self.view.set_size (1.0,.75)
                self.view.set_position (0,-1)
            self.reset()
    

    _window = Window (w,h,title + ', %s backend,' % backend ,fps)
    namespace['window'] = _window

    view = Viewport (name="view1", reset=_window.reset)
    view.append (Background())
    namespace['view'] = view
    _window.viewport.append (view)
    _window.view = view

    _window.terminal = Terminal (namespace, shell, _window.paint, _window.idle)
    _window.terminal.logo = logo

    _window.set_layout (layout)

    return _window

if __name__ == '__main__':
    win = window()
    win.show()    

