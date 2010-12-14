#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''

Show extra set of colormaps to be used with matplotlib
'''
from dana import *
from display import *

if __name__ == '__main__':
    Z = 1.25*gaussian((40,40), 0.10) - .75*gaussian((40,40), 0.50)

    fig = plt.figure(figsize=(10,10), facecolor='white')

    plot(plt.subplot(3,3,1), Z, 'ice-and-fire', cmap=ice_and_fire)
    plot(plt.subplot(3,3,2), Z, 'ice',          cmap=ice)
    plot(plt.subplot(3,3,3), Z, 'fire',         cmap=fire)
    plot(plt.subplot(3,3,4), Z, 'light-red',    cmap=light_red)
    plot(plt.subplot(3,3,5), Z, 'dark-red',     cmap=dark_red)
    plot(plt.subplot(3,3,6), Z, 'light-green',  cmap=light_green)
    plot(plt.subplot(3,3,7), Z, 'dark-green',   cmap=dark_green)
    plot(plt.subplot(3,3,8), Z, 'light-blue',   cmap=light_blue)
    plot(plt.subplot(3,3,9), Z, 'dark-blue',    cmap=dark_blue)

    plt.show()
