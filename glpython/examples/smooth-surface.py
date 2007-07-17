#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
#------------------------------------------------------------------------------

import scipy
from glpython import *

win,fig = window (size=(800,600), title = "smooth-surface.py")

def f(x, y):
    return scipy.sin(x*y+1e-4)/(x*y+1e-4)
x = scipy.arange(-5., 5., 0.2, dtype='float32')
y = scipy.arange(-5., 5., 0.2, dtype='float32')
X,Y = numpy.meshgrid(x, y)
Z = f(X, Y)

CM_IceAndFire.scale (-.5,1)
fig.smooth_surface (Z, cmap=CM_IceAndFire, zscale = .25)
fig.text (text="(sin(x*y+1e-4)/(x*y+1e-4)")
fig.colorbar(cmap=CM_IceAndFire)

win.show()
