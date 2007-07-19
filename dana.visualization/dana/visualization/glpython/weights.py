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
""" Weights view with a colorbar

"""

import OpenGL.GL as GL
from glpython import *

class Weights (Object):
    """ Show weights as several arrays """
    
    def __init__(self, src, dst, figure, cmap = CM_Default, style = 'flat',
                 title = None, show_colorbar = True, has_borde=True,
                 show_label = True):

        Object.__init__(self)
        self.src = src
        self.dst = dst
        self.visible = True
        self.active = True
        self.has_border = has_border
        self.maps = []

        # Overall size  
        w = dst.map.shape[0] * (src.map.shape[0]+1)+1
        h = dst.map.shape[1] * (src.map.shape[1]+1)+1

        self.sx = 1
        self.sy = 1
        if h > w:
            self.sy = float(h)/float(w)
        elif h < w:
            self.sx = float(w)/float(h)

        for unit in dst:
            frame = (
                (unit.position[0] * (src.map.shape[0]+1)+1)/float(w),
                (unit.position[1] * (src.map.shape[1]+1)+1)/float(h),
                (src.map.shape[0])/float(w),
                (src.map.shape[1])/float(h))
            if style == 'flat':
                array = FlatSurface (unit.weights(src), frame=frame)
            else:
                array = SmoothSurface (unit.weights(src), frame=frame)            
            self.maps.append ( (array, unit, src) )

        # Title
        self.title = Text (text='')
        if not title:
            self.title.visible = False
        else:
            self.title.text = title
        figure.append (self.title)

        # Colorbar
        self.colorbar = Colorbar(cmap=cmap)
        if not show_colorbar:
            self.colorbar.visible = False
        figure.append (self.colorbar)


    def render (self):
        """
        """

        GL.glPushMatrix()
        GL.glScalef (self.sx,self.sy,1)

        GL.glDisable (GL.GL_TEXTURE_RECTANGLE_ARB)
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        mode = GL.glGetIntegerv (GL.GL_RENDER_MODE)
        if mode == GL.GL_RENDER and self.has_border:
            GL.glColor3f (0,0,0)
            GL.glBegin (GL.GL_QUADS)
            GL.glVertex2f (-0.5, -0.5)
            GL.glVertex2f ( 0.5, -0.5)
            GL.glVertex2f ( 0.5,  0.5)
            GL.glVertex2f (-0.5,  0.5)
            GL.glEnd ()
        
        for array, unit, src in self.maps:
            array.data = unit.weights(src)
            array.render()
        GL.glPopMatrix()

