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
import Image as PIL
from glpython import *

win,fig = window (size=(800,600), title = "image.py")

im = PIL.open ('lena.png')
im = im.transpose (PIL.FLIP_TOP_BOTTOM)
ir,ig,ib = im.split()
r = numpy.asarray(ir, dtype='float32')
r = numpy.divide (r, 256.0)
g = numpy.asarray(ig, dtype='float32')
g = numpy.divide (g, 256.0)
b = numpy.asarray(ib, dtype='float32')
b = numpy.divide (b, 256.0)

fig.flat_surface ("lena.png", frame = (0.0, 0.5, 0.5, 0.5))
fig.flat_surface (r, frame = (0.0, 0.0, 0.5, 0.5), cmap=CM_Red)
fig.flat_surface (g, frame = (0.5, 0.0, 0.5, 0.5), cmap=CM_Green)
fig.flat_surface (b, frame = (0.5, 0.5, 0.5, 0.5), cmap=CM_Blue)

win.show()
