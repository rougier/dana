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

class Frame:
    def __init__(self, frame):
        self.frame = frame

    def render (self, cr):
        cr.move_to (0, 0) 
        cr.rel_line_to (0,                   self.frame.size[1])
        cr.rel_line_to (self.frame.size[0],  0)
        cr.rel_line_to (0,                  -self.frame.size[1])
        cr.rel_line_to (-self.frame.size[0], 0)
        cr.stroke()

