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

class Colorbar:

    def __init__(self, colorbar):
        self.colorbar = colorbar


    def render (self, cr):

        size = self.colorbar.size
        position = self.colorbar.position
        cmap = self.colorbar.cmap


        # Background
        gradient = cairo.LinearGradient (position[0], position[1],
                                         position[0], position[1]+size[1])
        n = 24
        for i in xrange (n):
            c = cmap.color(cmap.min + i/float(n) * (cmap.max-cmap.min))
            r,g,b = c.red, c.green, c.blue
            gradient.add_color_stop_rgb (1-i/float(n),  r,g,b)
        
        cr.rectangle (position[0], position[1],
                      size[0], size[1])
        cr.set_source (gradient)
        cr.fill ()

        # Frame
        cr.set_source_rgb (0,0,0)
        cr.set_line_width (max (cr.device_to_user_distance (1,1)))
        cr.rectangle (position[0], position[1],
                      size[0], size[1])
        cr.stroke ()
        
        # Ticks
        s = size[1]/self.colorbar.tick_number
        for i in xrange(int(self.colorbar.tick_number)):
            cr.move_to     (position[0], position[1]+(i+1)*s)
            cr.rel_line_to (.15*size[0],  0)
            cr.stroke()

            cr.move_to     (position[0]+.85*size[0], position[1]+(i+1)*s)
            cr.rel_line_to (.15*size[0],  0)
            cr.stroke()
        
        # Labels
        cr.select_font_face ("sans")
        cr.set_font_size (0.333/self.colorbar.tick_number)

        s = size[1]/self.colorbar.tick_number
        for i in xrange(int(self.colorbar.tick_number)+1):
            v = cmap.min + i/float(self.colorbar.tick_number) * (cmap.max-cmap.min)
            if v >=0:
                text = " +%.2f" % v
            else:
                text = " %.2f" % v
            x_bearing, y_bearing, width, height, x_advance,y_advance = \
                cr.text_extents (text)
            cr.move_to (position[0]+size[0], position[1]+size[1] - i*s + height/2.0 ) 
            cr.show_text (text)

