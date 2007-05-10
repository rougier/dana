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
""" Abstraction of an OpenGL object

An OpenGL object is an object that is renderable.

Attributes:
    visible - whether to render object or not
    active  - whether to update object internal state or not
    focus   - whether object has focus or not
    depth   - relative depth (rendering order)
"""


class Object:
    """ Abstraction of an OpenGL object """

    def __init__(self):
        """ Default initialization of the object """

        self.visible = True
        self.active = True
        self.focus = False
        self.depth = 1

    def init (self):
        """ Initialization"""
        pass

    def render (self):
        """ Rendering """
        pass

    def select (self, id1, id2):
        """ OpenGl: selection """
        pass

    def unselect (self):
        """ OpenGl: unselection """
        pass

    def resize (self, w, h):
        """ OpenGL: resize """
        pass

    def check_focus (self,x,y):
        """ Check if object has focus """
        pass

    def key_press (self, key):
        """ Key press event """
        pass

    def button_press (self, button, x,y):
        """ Mouse button press event """
        pass

    def button_release (self, x,y):
        """ Mouse button release event """
        pass

    def mouse_motion (self, x,y):
        """ Mouse motion event """
        pass


if __name__ == '__main__':
    print
    print """This is a pure abstract class and """ \
          """cannot be run in stand-alone mode."""
    print
