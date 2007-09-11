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


win,fig = window (size=(800,600), title = "shell.py",
                  has_terminal = True,
                  namespace = locals())

CM_Hot.scale (-.25,1)
fig.smooth_surface (Z, cmap=CM_Hot)
fig.text (text="(1-x/2+x**5+y**3)*exp(-x**2-y**2)")
fig.colorbar(cmap=CM_Hot)



win.show()
