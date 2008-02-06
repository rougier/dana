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
""" Reference frame object """

import gl, cairo


class Frame:
    """ Frame represents the outer frame of a figure """

    def __init__ (self, figure, size):
        """ Create a new Frame object  """

        self.visible = True
        self.figure = figure
        self.tick_length = 0.01
        self.tick_number = 10
        self.size = size
        self.renderer = {}
        self.renderer['opengl'] = gl.Frame (self)
        self.renderer['cairo']  = cairo.Frame (self)


    def render (self, renderer = 'opengl', data=None):
        """ Render using the specified renderer """

        if not self.visible:
            return
        elif renderer in self.renderer.keys():
            self.renderer[renderer].render (data)
