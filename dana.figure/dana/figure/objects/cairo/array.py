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

import cairo

class Array:

    def __init__(self, array):
        self.array = array

    def render (self, cr):
        w,h = self.array.figure.normalized_size
        size = self.array.size
        position = self.array.position
        cmap = self.array.cmap
        d0 = self.array.array.shape[0]
        d1 = self.array.array.shape[1]
        cr.set_line_width (max (cr.device_to_user_distance (1,1)))
        dx = size[0]/float(d1)
        dy = size[1]/float(d0)

        # Cells
        for i in range (d0):
            for j in range (d1):
                color = cmap.color (float(self.array.array[i][j]))
                cr.set_source_rgb (color.red, color.green, color.blue)
                cr.rectangle (position[0]+j*dx, h-position[1]-(i+1)*dy, dx, dy)
                cr.fill()

        # Frame
        cr.set_source_rgb (0,0,0)
        cr.rectangle (position[0], h-position[1], size[0], -size[1])
        cr.stroke ()

        #Name
        cr.select_font_face ("sans")
        cr.set_font_size (self.array.fontsize*0.333)
        cr.move_to (position[0]+0.05*size[0],
                    h - position[1]-0.05*size[1])
        cr.show_text (self.array.name)

