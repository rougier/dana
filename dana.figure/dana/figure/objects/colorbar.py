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

import gl,cairo

class Colorbar:

    def __init__ (self, figure, cmap):
        self.visible = True
        self.figure = figure
        self.position = (1.1*figure.normalized_size[0], 0)
        self.size = (0.05, 1.0*figure.normalized_size[1])
        self.cmap = cmap
        self.tick_number = 10.0
        self.renderer = {}
        self.renderer['opengl'] = gl.Colorbar (self)
        self.renderer['cairo']  = cairo.Colorbar (self)


    def render (self, renderer = 'opengl', data=None):
        if not self.visible:
            return
        elif renderer in self.renderer.keys():
            self.renderer[renderer].render (data)
