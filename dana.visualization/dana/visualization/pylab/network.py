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
# $Id: network.py 140 2007-04-14 08:38:41Z rougier $
#------------------------------------------------------------------------------
""" Interactive network view with/without colorbar 

    - mouse button 1 shows weight for the selected unit
    - mouse button 3 shows network activity

"""

from matplotlib.backend_bases import NavigationToolbar2 as toolbar
import matplotlib.pylab as pylab
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
import dana.core as core


class View (object):
    """ Network view with colorbar """

    def __init__ (self, network,
                  title='', use_colorbar=False, size=8, fontsize=20):
        """ Creation of the view """
        
        object.__init__(self)
        
        if use_colorbar:
            dx = 1.25
        else:
            dx = 1
        w,h = network.shape
        fig = pylab.figure (figsize= (size*dx, h/float(w)*size))
                
        pylab.connect ('button_press_event', self.on_click)
        data = {
            'red':   ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 0., 0.)),
            'blue':  ((0., 1., 1.), (.5, 1., 1.), (.75, 0., 0.), (1., 0., 0.))}
        cm = colors.LinearSegmentedColormap('cm',  data)

        self.network = network
        self.maps = []
        self.unit = None
        
        for m in network:
            x,y,w,h = m.frame
            axes = pylab.axes ((x/dx,y,w/dx,h))            
            axes.map = m
            if len(m) > 0:
                axes.data = m[0].potentials()
                im = pylab.imshow (axes.data, cmap=cm, vmin=-1.0, vmax=1.0,
                                   origin='upper', interpolation='nearest')
            if hasattr(m, 'name'):
                axes.text (0.5, 0.5, m.name, size=fontsize)
            pylab.title (title)
            pylab.setp(axes, xticks=[], yticks=[])
            self.maps.append( (m, axes, im) )
        
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

    def on_click (self, event):
        """
        Handler for mouse click events on a figure
        
        Mouse button 1 shows weights for the clicked unit
        Mouse button 3 shows all network activity
        """
        
        if event.inaxes:
            m = event.inaxes.map
            if event.button == 1:
                x = int(event.xdata)
                y = event.inaxes.data.shape[0]-1-int(event.ydata)
                self.unit = m[0].unit (x,y)
                self.update()           
            elif event.button == 3:
                self.unit = None
                self.update()
        return

    def update (self):
        """ Update view """
    
        if self.unit:
            for (m,axes,im) in self.maps:
                axes.data = self.unit.weights(m[0])
                im.set_data (axes.data)
        else:
            for (m,axes,im) in self.maps:
                axes.data = m[0].potentials()
                im.set_data (axes.data)
        pylab.draw()
        return


if __name__ == '__main__':
    import random
    import dana.core as core
    import dana.projection as projection
    import dana.projection.distance as distance
    import dana.projection.density as density
    import dana.projection.shape as shape
    import dana.projection.profile as profile
    
    net = core.Network()
    size = 20

    m0 = core.Map( (size, size), (0,0) )
    m0.append( core.Layer() )
    m0[0].fill(core.Unit)
    m0.name = "m0"
    net.append(m0)

    m1 = core.Map( (size, size), (1,1) )
    m1.append( core.Layer() )
    m1[0].fill(core.Unit)
    m1.name = "m1"
    net.append(m1)
        
    proj = projection.projection()
    proj.self = False
    proj.distance = distance.euclidean(False)
    proj.density = density.full(1)
    proj.shape = shape.box(1,1)
    proj.profile = profile.linear (0,1)
    proj.src = m0[0]
    proj.dst = m1[0]
    proj.connect()
    
    for u in m0[0]:
        u.potential = random.uniform (0,1)
    for u in m1[0]:
        u.potential = random.uniform (0,1)        
    
    view = View (net)
    view.show()
