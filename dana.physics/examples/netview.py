#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006, Nicolas Rougier.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under the
# conditions described in the aforementioned license. The license is also
# available online at http://www.loria.fr/~rougier/pub/Licenses/BSD.txt
# 
# Author: Nicolas Rougier
#------------------------------------------------------------------------------
""" Network view using matplotlib

This display a dana.core.network using matplotlib.
    mouse button 1 shows weight for the selected unit
    mouse button 3 shows network activity
"""

import matplotlib.pylab as pylab
import matplotlib.colors as colors
import dana.core as core


class view(object):
    """ Interactive dana.core.network view """

    def __init__(self, network, title='', size = 8):
        """ Initializes view """
        
        object.__init__(self)
        
        w,h = network.shape
        fig = pylab.figure (figsize= (size, h/float(w)*size))
        pylab.connect ('button_press_event', self.on_click)
        pylab.connect ('key_press_event', self.on_key)
        data = {
            'red':   ((0., 0., 0.), (.25, .75, .75), (.5, 1., 1.), (.75, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (.25, .75, .75), (.5, 1., 1.), (.75, 1., 1.), (1., 0., 0.)),
            'blue':  ((0., 1., 1.), (.25, 1., 1.), (.5, 1., 1.), (.75, 0., 0.), (1., 0., 0.))}
        cm = colors.LinearSegmentedColormap('cm',  data)
    
        for m in network:
            m.axes = pylab.axes (m.frame)
            m.axes.net = network
            m.axes.map = m
            if len(m) > 0:
                m.im = m.axes.imshow (m[0].potentials(), cmap=cm,
                                     vmin=-1.0, vmax=1.0,
                                     origin='lower', interpolation='nearest')
            if hasattr(m, 'name'):
                m.axes.text (0.5, 0.5, m.name, size=20)
            pylab.title (title)
            pylab.setp(m.axes, xticks=[], yticks=[])
        self.network = network
        self.network.view_unit = None
        return

    def show(self):
        """ Show figure """
        
        pylab.show()
        return

    def on_click (self, event):
        """
        Handler for mouse click events on a figure
        
        Mouse button 1 sets a positive source
        Mouse button 3 sets a negative source
        Mouse button 2 removes a source
        """
        
        if event.inaxes:
            net = event.inaxes.net
            map = event.inaxes.map
            unit = map[0].unit (int(event.xdata), int(event.ydata))
            if event.button == 1:
                unit.source = True
                unit.potential = 1
            elif event.button == 3:
                unit.source = True
                unit.potential = -1
            else:
                unit.source = False 
        return
        
    def on_key (self, event):
        if event.key == ' ' and event.inaxes:
            net = event.inaxes.net
            map = event.inaxes.map
            for u in map[0]:
                if not u.source:
                    u.potential = 0

    def update (self):
        """ Update view """
    
        unit = self.network.view_unit
        net = self.network
        
        if unit:
            for m in net:            
                m.im.set_data (unit.weights(m[0]))
            pylab.draw()
        else:
            for m in net:
                m.im.set_data (m[0].potentials())
            pylab.draw()
        return
  
