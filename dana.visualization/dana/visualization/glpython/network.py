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
""" GLPython network view

"""

import OpenGL.GL as GL
from glpython import *

class Network (Object):

    def __init__ (self, network, figure, cmap = CM_Default, style = 'flat',
                 title = None, show_colorbar = True, has_border=True,
                 show_label = True):
        """
        
        """
        
        Object.__init__(self)
        cmap.scale (-1,1)
        self.maps = []
        self.labels = []
        self.unit = None
        self.has_border = has_border
        self.network = network

        w, h = network.shape
        self.sx = 1
        self.sy = 1
        if h > w:
            self.sy = float(h)/float(w)
        elif h < w:
            self.sx = float(w)/float(h)

        for m in network:
            if style == 'flat':
                array = FlatSurface (m[0].potentials(),
                                     cmap=cmap, frame = m.frame)
            else:
                array = SmoothSurface (m[0].potentials(),
                                       cmap=cmap, frame = m.frame)            
            array.connect ('select_event', self.on_select, m)
            self.maps.append ((m, array))
            
            # Label
            name = ''
            if hasattr(m, 'name'):
                name = m.name
            label = Label (text=name,
                           position = (m.frame[0]+m.frame[2]/2.0-.5,
                                       m.frame[1]+m.frame[3]/2.0-.5,
                                       0, .5),
                           size = .075)
            if not name or not show_label:
                label.visible = False
            self.labels.append (label)
            figure.append (label)

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
        GL.glDisable(GL.GL_TEXTURE_RECTANGLE_ARB)
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

        self.update()
        for m, array in self.maps:
            array.render()
        GL.glPopMatrix()

    def select (self, x):
        """
        """
        self.unit = None
        for m,array in self.maps:
            array.select (x)

    def on_select (self, x, y, m):
        """
        """
        
        self.unit = m[0].unit (x,y)
        self.update()

    def update (self):
        """
        """
        if self.unit:
            for m,array in self.maps:
                array.data = self.unit.weights(m[0])
        else:
            for m,array in self.maps:
                array.data = m[0].potentials()

