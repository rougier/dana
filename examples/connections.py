#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
This example shows variopus connections and how to display them.
'''
from dana import *
from functools import partial

def format_coord(axis, x, y):
    Z = axis.get_array()
    if x is None or y is None or Z is None:
        return ''
    x,y = int(x), int(y)
    if 0 <= x < Z.shape[1] and 0 <= y < Z.shape[0]:
        return '[%d,%d]: %s' % (x,y, Z[y,x])
    return ''

def update(G, x, y):
    mgr = plt.get_current_fig_manager()
    if G is None:
        for axis,Z in mgr.subplots:
            axis.set_data (Z)
    else:
        for axis,z in mgr.subplots:
            axis.set_data(np.empty_like(z)*np.NaN)
        for C in G._connections:
            for axis,z in mgr.subplots:
                if C._actual_source is z:
                    axis.set_data(C[y,x])
    plt.draw()

def button_press_event(event):
    G,x,y = None, -1, -1
    if event.inaxes and event.button == 1:
        G = event.inaxes.group
        x,y = int(event.xdata), int(event.ydata)
    update(G, x, y)


def plot(subplot, group, data, name):
     mgr = plt.get_current_fig_manager()
     a,b = 0.75, 1.0
     chessboard = np.array(([a,b]*16 + [b,a]*16)*16)
     chessboard.shape = 32,32
     plt.imshow(chessboard, cmap=plt.cm.gray, interpolation='nearest',
                extent=[0,group.shape[0],0,group.shape[1]], vmin=0, vmax=1)
     plt.hold(True)
     axis = plt.imshow(data, interpolation='nearest', cmap= plt.cm.PuOr_r,
                       origin='lower', vmin=-1, vmax=1,
                       extent=[0,group.shape[0],0,group.shape[1]])
     subplot.format_coord = partial(format_coord, axis)
     subplot.group = group
     mgr.subplots.append((axis,data))
     subplot.text(2, 2, name, fontsize=24)


n = 20
p = 2*n+1
A, B = Group((n,n),'V'), Group((n,n),'V')
C, D = Group((n,n),'V'), Group((n/2,n/2),'V')

# Difference of Gaussians connections
K = 1.25*gaussian((p,p),0.1) - 0.75*gaussian((p,p),1.0)
DenseConnection(B,A,K)
SharedConnection(C,A,K)
SparseConnection(D,A,K)

# Row, columns and point connections
DenseConnection(A,B, np.ones((1,1)))
DenseConnection(C,B, np.ones((1,p)))
DenseConnection(D,B, np.ones((p,1)))

# Random connections
SparseConnection(A, C, rnd.random((p,p)) * (rnd.random((p,p)) > .8))
SparseConnection(B, C, rnd.random((n,n)) * (rnd.random((n,n)) > .8))
SparseConnection(D, C, rnd.random((n/2,n/2)) * (rnd.random((n/2,n/2)) > .8))

# Shifted connections
#SparseConnection(A,D,)
#SparseConnection(C,D,)
#SparseConnection(B,D,)

fig = plt.figure(figsize=(10,10))
mgr = plt.get_current_fig_manager()
mgr.subplots = []

plot(plt.subplot(2,2,1), A, A.V, 'A')
plt.title('Differenc of Gaussians using\n dense, shared and sparse connection')
plot(plt.subplot(2,2,2), B, B.V, 'B')
plt.title('Row, column and point connections')
plot(plt.subplot(2,2,3), C, C.V, 'C')
plt.title('Random sparse connections')
plot(plt.subplot(2,2,4), D, D.V, 'D')
plt.connect('button_press_event', button_press_event)
plt.show()
