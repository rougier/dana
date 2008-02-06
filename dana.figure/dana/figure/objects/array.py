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

import gl, cairo

class Array: 
    def __init__ (self, figure, array, cmap, position, size, fontsize, name= ""):
        self.figure = figure
        self.visible = True
        self.position = position
        self.size = size
        self.cmap = cmap
        self.array = array
        self.name = name
        self.fontsize = figure.fontsize
        self.renderer = {}
        self.renderer['opengl'] = gl.Array (self)
        self.renderer['cairo']  = cairo.Array (self)
        self.id = self.renderer['opengl'].id


    def render (self, renderer = 'opengl', userdata=None):
        if not self.visible:
            return
        if renderer in self.renderer.keys():
            self.renderer[renderer].render (userdata)

    def set_array (self, array):
        self.array = array
        self.renderer['opengl'].data = array
        self.id = self.renderer['opengl'].id
