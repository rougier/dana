#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------
""" Various GLPython objects

"""

import OpenGL.GL as GL
import glpython.core
from _objects import *
import os.path

class Brain (glpython.core.Object):
    """ Brain object
    
    The brain object is a mesh object that represents a brain
    """

    def __init__ (self, path, alpha=.25, scale=1):
        glpython.core.Object.__init__(self)
        
        self.scale = scale
        self.alpha = alpha
        self.brain = Model (
            os.path.join (path, "brain.3ds"), (1,1,0), alpha)

    def render(self):
        GL.glPushMatrix ()
        s = self.scale
        GL.glScalef (s,s,s)
        self.brain.render()
        GL.glPopMatrix()

__all__ = ['Cube', 'Background', 'Colorbar', 'Text', 'Label', 'Model',
           'Array', 'FlatSurface', 'SmoothSurface', 'CubicSurface', 'Brain']

