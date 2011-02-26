#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
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
