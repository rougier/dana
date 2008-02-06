#!/usr/bin/env python
# ------------------------------------------------------------------------------
# DANA -- Distributed Asynchronous Numerical Adaptive computing library
# Copyright (c) 2007  Nicolas P. Rougier
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
# ------------------------------------------------------------------------------
""" visualization

    A Figure offers visualization to numpy arrays using either RGB format for
    (n,m,3) shaped arrays or using a colorscale for (n,m) shaped arrays. The
    figure can be rendered onscreen using the dedicated gtk-opengl backend or
    it can be directly saved into a filename (or both).
    
    Example:
       >>> from dana.figure import *
       >>> figure = Figure()
       >>> figure.save ('figure.pdf')
       >>> figure.show()

    If you choose to visualize on screen, the 'show' command will enter the
    backend mainloop and thus, you won't be able to interact anymore. If you
    want to interact while visualizing onscreen, it is your responsability to
    run a mainloop on your side. For example, you can use:

       % ipython -wthread
"""

from _figure import *
import _figure
from backend_wx import Backend
from objects import Array, Frame, Colorbar
import cairo



class Figure:
    """ Figure class """

    def __init__ (self, network, fontsize=.15):
        """ Create a new empty figure of given size (inch) """
        
        w,h = network.shape[0]*8.0, network.shape[1]*8.0
        if w < h:
            self.normalized_size = (float(w)/float(h), 1.0)
        else:
            self.normalized_size = (1.0, float(h)/float(w))
        self.size = self.normalized_size[0] * 8.0, self.normalized_size[1] * 8.0
        self.zoom = 1.0
        self.position = 0,0
        W,H = self.normalized_size
        self.backend = None
        self.fontsize = fontsize
        self.frame = Frame (self, self.normalized_size)
        self.cmap = _figure.CM_IceAndFire
        self.cmap.scale(-1,1)
        self.colorbar = Colorbar (self, self.cmap)
        self.selected_unit = None
        fontsize = .15
        self.maps = []
        self.network = network
        for m in network:
            W,H = self.normalized_size
            x,y,w,h = m.frame[0]*W, m.frame[1]*H, m.frame[2]*W, m.frame[3]*H 
            name = ""
            if hasattr(m, 'name'):
                name = m.name
            Z = m[0].potentials
            self.maps.append ([m, Array (self, Z, self.cmap, (x,y), (w,h), fontsize, name)])
            
    def reset (self):
        """ Reset figure """

        if self.position != (0,0):
            self.position = (0,0)
            return True
        elif self.zoom != 1.0:
            self.zoom = 1.0
            return True
        return False


    def render (self, renderer = 'opengl', data = None):
        """ Draw figure """

        self.update()
        for map, array in self.maps:
            array.render (renderer, data)
        self.frame.render (renderer, data)
        self.colorbar.render (renderer, data)


    def show (self):
        """ Show figure """

        if not self.backend:
            self.backend = Backend (self)
            self.backend._select_callback = self.select
        self.backend.show ()


    def select (self, id):
        for map, array in self.maps:
            if (id >= array.id and
                id < (array.id + array.array.shape[0]*array.array.shape[1])):
                x = (id-array.id)%array.array.shape[0]
                y = (id-array.id)/array.array.shape[0]
                self.selected_unit = map[0].unit(x,y)
                return
        self.selected_unit = None

    def hide (self):
        """ Hide figure """

        if self.backend:
            self.backend.hide ()


    def update (self):
        """  """
        if self.selected_unit:
            for m,array in self.maps:
                array.set_array (self.selected_unit.weights(m[0]))
        else:
            for m,array in self.maps:
                array.set_array (m[0].potentials)


    def save (self, filename):
        """ Save figure into specified filename (pdf or png) """

        if self.backend:
            w, h = self.backend.size
        else:
            w, h = self.size[0]*72, self.size[1]*72

        size = max (w,h)/1.0
        size *= self.zoom
        width  = w/size
        height = h/size
        dx = self.position[0]
        dy = self.position[1]

        ext = filename.split('.')[-1]
        if not ext:
            filename.append ('pdf')
            ext = 'pdf'

        if ext == 'pdf':
            surface = cairo.PDFSurface (filename, w, h)
            cr = cairo.Context (surface)
            cr.scale (size,size)
            cr.translate ((width-self.normalized_size[0])/2.0 + dx,
                          (height-self.normalized_size[1])/2.0 + dy)
            self.render ('cairo', cr)
            del surface
        elif ext == 'png':
            surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, int(w), int(h))
            cr = cairo.Context (surface)
            cr.scale (size,size)
            cr.translate ((width-self.normalized_size[0])/2.0 + dx,
                          (height-self.normalized_size[1])/2.0 + dy)
            self.render ('cairo', cr)
            surface.write_to_png (filename)
            del surface        

