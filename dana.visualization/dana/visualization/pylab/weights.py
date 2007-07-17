#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: weights.py 139 2007-04-14 08:37:31Z rougier $
#------------------------------------------------------------------------------
""" Weights view with a colorbar

"""

from matplotlib.backend_bases import NavigationToolbar2 as toolbar
import matplotlib.pylab as pylab
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
import dana.core as core

class View (object):
    """ Weights view with a colorbar """    
    
    def __init__(self, layer, source,
                 title='', use_colorbar=False, size=8, fontsize=20):
        """ Create the figure for layer using weights coming from source """

        object.__init__(self)
        self.source = source

        # Overall size  
        w = layer.map.shape[0] * (source.map.shape[0]+1)+1
        h = layer.map.shape[1] * (source.map.shape[1]+1)+1
        
        if use_colorbar:
            dx = 1.25
        else:
            dx = 1
        
        # Create new figure
        if h<w:
            fig = pylab.figure (figsize= (size*dx, h/float(w)*size))
        else:
            fig = pylab.figure (figsize= (w/float(h)*size*dx, size))

        # Colormap
        data = {
            'red':   ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 0., 0.)),
            'blue':  ((0., 1., 1.), (.5, 1., 1.), (.75, 0., 0.), (1., 0., 0.))}
        cm = colors.LinearSegmentedColormap('cm',  data)

        # Creation of axes (one per unit)
        self.units = []
        for unit in layer:
            frame = (
                ((unit.position[0] * (source.map.shape[0]+1)+1)/float(w))/dx,
                (unit.position[1] * (source.map.shape[1]+1)+1)/float(h),
                ((source.map.shape[0])/float(w))/dx,
                (source.map.shape[1])/float(h))
            axes = pylab.axes(frame)
            axes.unit = unit
            axes.data = unit.weights(source)
            im = axes.imshow(axes.data, cmap=cm, vmin=-1.0, vmax=1.0,
                             origin='lower', interpolation='nearest')
            pylab.setp(axes, xticks=[], yticks=[])
            self.units.append ( (unit, axes, im) )

        if use_colorbar:
            axes = pylab.axes ( (0.80, 0.1, .25, .8) )
            axes.axis("off")
            pylab.title("Activity levels")
            cax, kw = colorbar.make_axes(axes, fraction=1/dx,
                                         pad=-0.5, aspect = 20)
            c = colorbar.ColorbarBase(ax=cax, cmap=cm,
                                      norm=colors.normalize (-1,1))
        return

    def show(self):
        """ Show figure """
        
        pylab.show()
        return
        
    def update(self):
        """ Update weights """
        
        for unit, axes, im in self.units:
            axes.data = unit.weights(self.source)
            im.set_data (axes.data)
        pylab_draw()

