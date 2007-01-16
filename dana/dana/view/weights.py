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
""" map weights visualization

"""

from matplotlib.backend_bases import NavigationToolbar2 as toolbar
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import dana.core as core


def mouse_move(self, event):
    """
    Call back on the mouse move event within a figure.
    
    This function replaces the original mouse_move function of the
    generic NavigationToolbar2 in order to write some useful information
    within the navigation toolbar when the mouse is moving.

    """

    if (event.inaxes and event.inaxes.get_navigate() and
        self._active not in ('ZOOM', 'PAN')):
        try:
            unit = event.inaxes.unit
            x,y = int(event.xdata), int(event.ydata)
            s = "Weight from unit[%d,%d] to unit[%d,%d] : %.3f "  % \
               (x,y,unit.position[0], unit.position[1], event.inaxes.data[y,x])
        except ValueError:
            pass
        except OverflowError:
            pass
        else:
            self.set_message(s)
    else:
        self._mouse_move (event)
toolbar._mouse_move = toolbar.mouse_move
toolbar.mouse_move = mouse_move



class WeightsView (object):
    """ Visualization of weights from one layer to another
    
    """    
    
    def __init__(self, src, dst, size = 8):
        """ Creation of the figure """

        object.__init__(self)
    
        if isinstance (src, core.Map):
            src = src[0]
        if isinstance (dst, core.Map):
            dst = dst[0]

        # Overall size  
        w = dst.map.shape[0] * (src.map.shape[0]+1)+1
        h = dst.map.shape[1] * (src.map.shape[1]+1)+1
        
        # Create new figure
        if h<w:
            fig = pylab.figure (figsize= (size, h/float(w)*size))
        else:
            fig = pylab.figure (figsize= (w/float(h)*size, size))

        # Fetish colormap
        data = {
            'red':   ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 0., 0.)),
            'blue':  ((0., 1., 1.), (.5, 1., 1.), (.75, 0., 0.), (1., 0., 0.))}
        cm = colors.LinearSegmentedColormap('cm',  data)

        # Creation of axes, one per unit
        for unit in dst:
            frame = ((unit.position[0] * (src.map.shape[0]+1)+1)/float(w),
                     (unit.position[1] * (src.map.shape[1]+1)+1)/float(h),
                     (src.map.shape[0])/float(w),
                     (src.map.shape[1])/float(h))
            axes = pylab.axes(frame)
            axes.unit = unit
            axes.data = unit.weights(src)
            axes.imshow(axes.data, cmap=cm, vmin=-1.0, vmax=1.0,
                        origin='lower', interpolation='nearest')
            pylab.setp(axes, xticks=[], yticks=[])
        return

    def show(self):
        """ Show figure """
        
        pylab.show()


if __name__ == '__main__':
    import dana.core as core
    import dana.projection as projection
    import dana.projection.distance as distance
    import dana.projection.density as density
    import dana.projection.shape as shape
    import dana.projection.profile as profile
    
    net = core.Network()

    m0 = core.Map( (10,10), (0,0) )
    m0.append( core.Layer() )
    m0[0].fill(core.Unit)
    net.append(m0)
    
    m1 = core.Map( (5, 5), (0,0) )
    m1.append( core.Layer() )
    m1[0].fill(core.Unit)
    net.append(m1)
    
    proj = projection.projection()
    proj.self = False
    proj.distance = distance.euclidean(False)
    proj.density = density.full(1)
    proj.shape = shape.box(1,1)
    proj.profile = profile.uniform(0.0, 1.0)
    proj.src = m1[0]
    proj.dst = m0[0]
    proj.connect()
    
    figure = WeightsView (m1[0], m0[0])
    figure.show()

