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

import numpy
from glpython import *

def func3(x,y):
    return (1- x/2 + x**5 + y**3)*numpy.exp(-x**2-y**2)

dx, dy = 0.05, 0.05
x = numpy.arange(-3.0, 3.0, dx,dtype='float32')
y = numpy.arange(-3.0, 3.0, dy,dtype='float32')
X,Y = numpy.meshgrid(x, y)
Z = func3(X, Y)

win,fig1 = window (size=(800,600), title = "surface.py")


CM_Hot.scale (-.25,1)
fig1.smooth_surface (Z, cmap=CM_Hot)
fig1.colorbar (position = (.1,.1), cmap=CM_Hot)
fig1.text (text='main figure', position = (.5,.1))

BG_cmap = Colormap()
BG_cmap.append (0, (1,1,1,1))
BG_cmap.append (1, (0,0,0,1))

CM_Ice.scale (-.25,1)
fig2 = fig1.figure (position = (-1,-1), size = (.5,.5))
fig2.flat_surface (Z, cmap=CM_Ice)
fig2.is_ortho = True
fig2.set_view (0,0,.65)
fig2.colorbar (cmap=CM_Ice)
fig2.text (text='sub figure')
fig2.background (cmap=BG_cmap, alpha=.5)

win.show()
