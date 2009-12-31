#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#   ____  _____ _____ _____ 
#  |    \|  _  |   | |  _  |   DANA, Distributed Asynchronous Adaptive Numerical
#  |  |  |     | | | |     |         Computing Framework
#  |____/|__|__|_|___|__|__|         Copyright (C) 2009 INRIA  -  CORTEX Project
#                         
#  This program is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free Software
#  Foundation, either  version 3 of the  License, or (at your  option) any later
#  version.
# 
#  This program is  distributed in the hope that it will  be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public  License for  more
#  details.
# 
#  You should have received a copy  of the GNU General Public License along with
#  this program. If not, see <http://www.gnu.org/licenses/>.
# 
#  Contact: 
# 
#      CORTEX Project - INRIA
#      INRIA Lorraine, 
#      Campus Scientifique, BP 239
#      54506 VANDOEUVRE-LES-NANCY CEDEX 
#      FRANCE
# 
"""
"""
from functools import partial
import numpy
import pylab

class view(object):

    def __init__ (self, items, size=5, fontsize=16, origin='lower',
                  cmap=pylab.cm.PuOr_r, interpolation = 'nearest',
                  vmin = -1, vmax = 1):
        """

        Parameters
        ----------

        items : [array/group, list of array/group, list of list of array/group]
            Arrays to be displayed

        size : float
            Figure inches size

        fontsize : float
            Text Font size

        cmap : [ None | Colormap ]
            A matplotib colormap` instance, eg. cm.jet.

        origin: [ None | 'upper' | 'lower' ]
            Place the [0,0] index of arrays in the upper left or lower left
            corner of the axes.

        vmin/vmax : [ None | scalar ]
            Used to scale a luminance image to 0-1.  If either is *None*, the
            min and max of the luminance values will be used.  Note if *norm* is
            not *None*, the settings for *vmin* and *vmax* will be ignored.

        interpolation : [ None | 'nearest' | 'bilinear' | 'bicubic' |
                         'spline16' | 'spline36' | 'hanning' | 'hamming' |
                         'hermite' | 'kaiser' | 'quadric' | 'catrom' |
                         'gaussian' | 'bessel' | 'mitchell' | 'sinc' |
                         'lanczos' | 'blackman' ]
        """

        rows,cols, Zi = 0, 0, []
        if isinstance (items, numpy.ndarray):
            rows,cols = 1, 1
            Zi.append ([(items,''), 1])
        elif isinstance (items, list):
            s = [item for item in items if isinstance(item,list)]
            if not s:
                cols, rows = len(items), 1
                for i in range (len(items)):
                    if isinstance (items[i], tuple):
                        Zi.append ( [items[i],i+1] )
                    else:
                        name = ''
                        if hasattr(items[i], 'name'):
                            name = items[i].name
                        Zi.append ( [(items[i],name),i+1] )
            else:
                rows, cols = len(items), max(len(item) for item in s)
                for i in range(len(items)):
                    if isinstance(items[i],list):
                        for j in range (len(items[i])):
                            if isinstance (items[i][j], tuple):
                                Zi.append ( [items[i][j],j+cols*i+1] )
                            else:
                                name = ''
                                if hasattr(items[i][j], 'name'):
                                    name = items[i][j].name
                                Zi.append ( [(items[i][j],name),j+cols*i+1] )
                    else:
                    	if isinstance (items[i], tuple):
                            Zi.append ([items[i],cols*i+1] )
                        else:
                            name = ''
                            if hasattr(items[i], 'name'):
                                name = items[i].name
                            Zi.append ( [(items[i],name),cols*i+1] )

        fig = pylab.figure( figsize=(cols*size,rows*size*(1.0/1.0)) )
        pylab.connect( 'button_press_event', self.button_press_event )
        #cmap = pylab.cm.PuOr_r
        self.subplots = []
        self.selection = None
        for Z, i in Zi:
            subplot = pylab.subplot(rows, cols, i)
            #pylab.grid(True)
            pylab.xticks([])
            pylab.yticks([])
            subplot.Z = Z[0]
            subplot.background = None
            axis = pylab.imshow( Z[0], vmin=vmin, vmax=vmax, cmap=cmap,
                                 origin=origin, interpolation=interpolation,
                                 extent=(0, Z[0].shape[1], 0,Z[0].shape[0]) )
            subplot.format_coord = partial (self.format_coord, axis)
            subplot.text( 0.0, 1.01, Z[1], fontsize=fontsize,
                          transform = subplot.transAxes )
            self.subplots.append( [axis, Z[0]] )

        # Colorbar on bottom
        fig.subplots_adjust(bottom=0.15, left=0.05, top=0.9, right=0.95)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.05])
        pylab.colorbar(cax=ax, orientation='horizontal')

        # Update
        self.update()

    def format_coord( self, axis, x, y):
        """ Return position/value as a string """

        Z = axis.get_array()
        if x is None or y is None or Z is None:
            return ''
        x,y = int(x), int(y)
        if 0 <= x < Z.shape[1] and 0 <= y < Z.shape[0]:
            return '[%d,%d]: %s' % (x,y, Z[y,x])
        return ''

    def button_press_event( self, event ):
        if not event.inaxes:
            self.selection = None
            self.update()
            return          
        Z = event.inaxes.Z
        x,y = int(event.xdata), int(event.ydata)
        if not event.inaxes or event.button != 1:
            self.selection = None
            return
        self.selection = Z,x,y
        self.update()
        return

    def update (self):
        """ Update all data """

        if not self.selection:
            for axis,Z in self.subplots:
                axis.set_data (Z)
        else:
            Z,x,y = self.selection
            for axis,z in self.subplots:
                W = Z.parent.get_weight(z,(y,x))
                if W is not None:
                    axis.set_data( W )
                #else:
                #    Z = axis.get_data(W)
                #    Z = numpy.ones_like(W)*numpy.NaN
                #    axis.set_data( W )
        pylab.draw()

    def show(self):
        """ Show figure and enter event loop """
        pylab.show()

