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

from glpython import *

win,fig = window (size=(800,600), title = "text.py")

fig.text (text = "Right justified text",
          position = (.5, -.1), alignment = 'right')

fig.text (text = "Centered text",
          position = (.5, -.2), alignment = 'center')

fig.text (text = "Left justified text",
          position = (.5, -.3), alignment = 'left')

fig.text (text = "Fixed size text",
          position = (.5, -.4), size = 24)

fig.text (text = "Relative size text",
          position = (.5, -.5), size = .2)

fig.text (text = "Vertical text",
          position = (.1, .5), orientation=90)

for i in range(20):
    fig.text (text = "          Text",
              position = (.5, -.8),  orientation=i*18, alignment = 'left',
              color = (i/20.0,i/20.0,i/20.0))

win.show()
