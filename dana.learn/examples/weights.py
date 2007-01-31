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
# $Id$
#------------------------------------------------------------------------------
""" Weights view

"""

from matplotlib.backend_bases import NavigationToolbar2 as toolbar
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import dana.core as core

class WeightsView (object):
    """ Weights view """    
    
    def __init__(self, layer, source, size = 8):
        """ Create the figure for layer using weights coming from source """

        object.__init__(self)
        self.source = source

        # Overall size  
        w = layer.map.shape[0] * (source.map.shape[0]+1)+1
        h = layer.map.shape[1] * (source.map.shape[1]+1)+1
        
        # Create new figure
        if h<w:
            fig = pylab.figure (figsize= (size, h/float(w)*size))
        else:
            fig = pylab.figure (figsize= (w/float(h)*size, size))

        # Colormap
        data = {
            'red':   ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 0., 0.)),
            'blue':  ((0., 1., 1.), (.5, 1., 1.), (.75, 0., 0.), (1., 0., 0.))}
        cm = colors.LinearSegmentedColormap('cm',  data)

        # Creation of axes (one per unit)
        self.units = []
        for unit in layer:
            frame = ( ((unit.position[0] * (source.map.shape[0]+1)+1)/float(w)),
                      (unit.position[1] * (source.map.shape[1]+1)+1)/float(h),
                      ((source.map.shape[0])/float(w)),
                      (source.map.shape[1])/float(h))
            axes = pylab.axes(frame)
            axes.unit = unit
            axes.data = unit.weights(source)
            im = axes.imshow(axes.data, cmap=cm, vmin=-1.0, vmax=1.0,
                             origin='lower', interpolation='nearest')
            pylab.setp(axes, xticks=[], yticks=[])
            self.units.append ( (unit, axes, im) )
	
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
        pylab.draw()
