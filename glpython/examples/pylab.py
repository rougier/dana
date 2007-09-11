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

import glpython
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import Image as PIL


fig = Figure()
dpi = fig.get_dpi()
fig.set_figwidth  (512.0/dpi)
fig.set_figheight (512.0/dpi)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax.plot([1,2,3])
ax.set_title('hi mom')
ax.grid(True)
ax.set_xlabel('time')
ax.set_ylabel('volts')
canvas.draw()
size = canvas.get_width_height()

buf = canvas.buffer_rgba(0,0)
im = PIL.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
#buf = canvas.tostring_rgb()
#im = PIL.fromstring('RGB', size, buf, 'raw', 'RGB', 0, 1)


win,fig = glpython.window ()
fig.image (im, frame = (0, 0, 1, 1))
win.show()
