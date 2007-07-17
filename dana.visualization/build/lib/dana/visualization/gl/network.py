#!/usr/bin/env python

#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: network.py 160 2007-05-11 12:50:33Z rougier $
#------------------------------------------------------------------------------
""" OpenGL view of a network

"""


from OpenGL.GL import *
from glpython.core import *
from glpython.objects import *
#from dana.visualization.gl import ArrayBar
#from dana.visualization.gl import Array

class View (Object):

    def __init__ (self, network, style = 'flat', fontsize=24):
        Object.__init__(self)
        self.maps = []
        self.unit = None
        self.network = network

        w, h = network.shape
        self.sx = 1
        self.sy = 1
        if h > w:
            self.sy = float(h)/float(w)
        elif h < w:
            self.sx = float(w)/float(h)

        for m in network:
            name = ''
            if hasattr(m, 'name'):
                name = m.name
            array = Image (m[0].potentials(), frame = m.frame)
            array.connect ('select_event', self.on_select, m)
            self.maps.append ((m, array))


    def render (self):
        glPushMatrix()
        glScalef (self.sx,self.sy,1)
        glDisable(GL_TEXTURE_RECTANGLE_ARB)
        glDisable(GL_LIGHTING)
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE)
        glColor3f (0,0,0)
#        glBegin (GL_QUADS)
#        glVertex2f (-0.5, -0.5)
#        glVertex2f ( 0.5, -0.5)
#        glVertex2f ( 0.5,  0.5)
#        glVertex2f (-0.5,  0.5)
#        glEnd ()
        self.update()
        for m, array in self.maps:
            array.render()
        glPopMatrix()

    def select (self, x):
        self.unit = None
        for m,array in self.maps:
            array.select (x)

    def on_select (self, x, y, m):
        self.unit = m[0].unit (x,m.shape[1]-1-y)
        self.update()

    def update (self):
        if self.unit:
            for m,array in self.maps:
                array.data = self.unit.weights(m[0])
        else:
            for m,array in self.maps:
                array.data = m[0].potentials()

