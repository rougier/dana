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
This example shows how to visualize connections from a group.
'''
from dana import *
from functools import partial

subplots = []

def format_coord(axis, x, y):
    ''' '''
    Z = axis.get_array()
    if x is None or y is None or Z is None:
        return ''
    x,y = int(x), int(y)
    if 0 <= x < Z.shape[1] and 0 <= y < Z.shape[0]:
        return '[%d,%d]: %s' % (x,y, Z[y,x])
    return ''

def update(G, x, y):
    ''' '''
    if G is None:
        for axis,Z in subplots:
            axis.set_data (Z)
    else:
        for axis,z in subplots:
            axis.set_data(np.empty_like(z)*np.NaN)
        for C in G._connections:
            for axis,z in subplots:
                if C._actual_source is z:
                    axis.set_data(C[y,x])
    plt.draw()

def button_press_event(event):
    ''' '''
    G,x,y = None, -1, -1
    if event.inaxes and event.button == 1:
        G = event.inaxes.group
        x,y = int(event.xdata), int(event.ydata)
    update(G, x, y)


    
n = 40
S = Group((n,n), 'V')
T = Group((n,n), 'V')
K = 1.25*gaussian((2*n+1,2*n+1),0.1) - 0.75*gaussian((2*n+1,2*n+1),1.0)
C = DenseConnection(S,T,K)

chessboard = np.array(([0.75,1]*16 + [1,0.75]*16)*16);
chessboard.shape = 32,32


plt.figure(figsize=(12,6))
cmap = cmap= plt.cm.PuOr

subplot = plt.subplot(1,2,1)
plt.imshow(chessboard, cmap=plt.cm.gray, interpolation='nearest',
           extent=[0,n,0,n], vmin=-1, vmax=2)
plt.hold(True)
axis = plt.imshow(S.V, interpolation='nearest', cmap= plt.cm.PuOr,
                  origin='lower', vmin=-1, vmax=1, extent=[0,n,0,n])
subplot.format_coord = partial(format_coord, axis)
subplot.group = S
subplots.append((axis,S.V))
subplot.text(2,2,'Source', fontsize=24)

subplot = plt.subplot(1,2,2)
plt.imshow(chessboard, cmap=plt.cm.gray, interpolation='nearest',
           extent=[0,n,0,n], vmin=0, vmax=1)
plt.hold(True)
axis = plt.imshow(T.V, interpolation='nearest', cmap= plt.cm.PuOr,
                  origin='lower', vmin=-1, vmax=1, extent=[0,n,0,n])
subplot.format_coord = partial(format_coord, axis)
subplot.group = T
subplots.append((axis,T.V))
subplot.text(2,2,'Target', fontsize=24)
plt.title('Click anywhere on this group')


plt.connect('button_press_event', button_press_event)
plt.show()
