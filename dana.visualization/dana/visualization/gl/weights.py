#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: weights.py 145 2007-05-10 14:18:42Z rougier $
#------------------------------------------------------------------------------
""" Weights view with a colorbar

"""

from OpenGL.GL import *
from glpython.object import Object
from dana.visualization.gl import Array
from dana.visualization.gl import ArrayBar

class View (Object):
    """ Show weights as several arrays """
    
    def __init__(self, layer, source, style = 'flat', fontsize=24):

        Object.__init__(self)
        self.source = source
        self.visible = True
        self.active = True
        self.maps = []

        MyArray = Array
        if style == 'bar':
            MyArray = ArrayBar

        # Overall size  
        w = layer.map.shape[0] * (source.map.shape[0]+1)+1
        h = layer.map.shape[1] * (source.map.shape[1]+1)+1

        self.sx = 1
        self.sy = 1
        if h > w:
            self.sy = float(h)/float(w)
        elif h < w:
            self.sx = float(w)/float(h)

        for unit in layer:
            frame = (
                (unit.position[0] * (source.map.shape[0]+1)+1)/float(w),
                ((source.map.shape[1]-1-unit.position[1]) * (source.map.shape[1]+1)+1)/float(h),
                (source.map.shape[0])/float(w),
                (source.map.shape[1])/float(h))
            array = MyArray (unit.weights(source), frame, '', fontsize)
            self.maps.append ( (array, unit, source) )


    def init (self):
        pass


    def render (self):
        """ """

        glPushMatrix()
        glScalef (self.sx,self.sy,1)

        glDisable(GL_TEXTURE_RECTANGLE_ARB)
        glDisable(GL_LIGHTING)
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE)
        glColor3f (0,0,0,0)
        glBegin (GL_QUADS)
        glVertex2f (-0.5, -0.5);
        glVertex2f ( 0.5, -0.5);
        glVertex2f ( 0.5,  0.5);
        glVertex2f (-0.5,  0.5);
        glEnd ();
        
        for array, unit, source in self.maps:
            array.set_data (unit.weights(source))
            array.render()
        glPopMatrix()
