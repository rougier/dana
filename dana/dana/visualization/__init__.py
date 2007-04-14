#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007,2006 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------
""" visualization tools

"""
from matplotlib.backend_bases import NavigationToolbar2 as toolbar
import dana.visualization.network
import dana.visualization.weights

def mouse_move(self, event):
    """
    Call back on the mouse move event within a figure.
    
    This function replaces the original mouse_move function of the
    generic NavigationToolbar2 in order to write some useful information
    within the navigation toolbar when the mouse is moving.

    """

    if (event.inaxes and event.inaxes.get_navigate() and
        self._active not in ('ZOOM', 'PAN')):

        if hasattr(event.inaxes, 'unit'):
            try:
                unit = event.inaxes.unit
                x = int(event.xdata)
                y = event.inaxes.data.shape[0]-1-int(event.ydata)
                s = "Weight from unit[%d,%d] to unit[%d,%d] : %.3f "  % \
                   (x,y,unit.position[0], unit.position[1],
                                        event.inaxes.data[y,x])
            except ValueError:
                pass
            except OverflowError:
                pass
            else:
                self.set_message(s)

        elif hasattr(event.inaxes, 'map'):
            try:
                x = int(event.xdata)
                y = event.inaxes.data.shape[0]-1-int(event.ydata)
                s = "[%d,%d] : %.3f " % (x,y,event.inaxes.data[y,x])
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




def View2D (object=None, source=None,
            title='', use_colorbar=False, size=8, fontsize=20):
    """ 2D view of a network or weights between layers
    
    """
    
    if object and not source:
        return dana.visualization.network.View2D (
            network=object,
            title=title, size=size, fontsize=fontsize,
            use_colorbar=use_colorbar)
    elif object and source:
        return dana.visualization.weights.View2D (
            layer=object, source=source,
            title=title, size=size, fontsize=fontsize,
            use_colorbar=use_colorbar)



