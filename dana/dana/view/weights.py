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
# $Id: netview.py 47 2007-01-13 11:51:16Z rougier $
#------------------------------------------------------------------------------
""" Network view using matplotlib

This display a dana.core.network using matplotlib.
    mouse button 1 shows weight for the selected unit
    mouse button 3 shows network activity
"""

import matplotlib.pylab as pylab
import matplotlib.colors as colors
import dana.core as core


class Weights (object):
    """ Weights view """

    def __init__(self, src, dst, size = 5):
        
        object.__init__(self)
        w,h = src.shape
        fig = pylab.figure (figsize= (size, h/float(w)*size))

        data = {
            'red':   ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 1., 1.)),
            'green': ((0., 0., 0.), (.5, 1., 1.), (.75, 1., 1.), (1., 0., 0.)),
            'blue':  ((0., 1., 1.), (.5, 1., 1.), (.75, 0., 0.), (1., 0., 0.))}
        cm = colors.LinearSegmentedColormap('cm', data)

        w = src.shape[0] * (dst.shape[0]+1)+1
        h = src.shape[1] * (dst.shape[1]+1)+1

        for u in src[0]:
            frame = ((u.position[0] * (src.shape[0]+1)+1)/float(w),
                     (u.position[1] * (src.shape[1]+1)+1)/float(h),
                     (src.shape[0])/float(w),
                     (src.shape[1])/float(h))
            axes = pylab.axes (frame)
            axes.imshow (u.weights(src[0]), cmap=cm, vmin=-1.0, vmax=1.0,
                         origin='lower', interpolation='nearest')
            pylab.setp(axes, xticks=[], yticks=[])
        return

    def show(self):
        pylab.show()
