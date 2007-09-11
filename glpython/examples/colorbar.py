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

win,fig = window (size=(800,600), title = "colorbar.py")

user_cmap = Colormap()
user_cmap.append (0.0,  (1,0,0))
user_cmap.append (0.5,  (0,1,0))
user_cmap.append (1.0,  (0,0,1))

cmaps     =  [CM_Ice, CM_Fire, CM_IceAndFire, CM_Hot,
              CM_Grey, CM_Red, CM_Green, CM_Blue, user_cmap]
cmap_names = ['CM_Ice', 'CM_Fire', 'CM_IceAndFire', 'CM_Hot',
              'CM_Grey', 'CM_Red', 'CM_Green', 'CM_Blue', 'user_cmap']

# Horizontal
for i in range(len(cmaps)):
    fig.colorbar (cmap=cmaps[i], size = .8, orientation = 'horizontal',
                  aspect = 30, position = (.1, (i+.5)/(len(cmaps)+.5)),
                  title = cmap_names[i])

# Vertical
#for i in range(len(cmaps)):
#    fig.colorbar (cmap=cmaps[i], size = .8, orientation = 'vertical',
#                  aspect = 25, position = ((i+.5)/(len(cmaps)+.5), .1),
#                  title = cmap_names[i])

win.show()
