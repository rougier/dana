#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009,2010,2011 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
from dana import *


n = 50
k = 2.5
G = zeros((n,n), 'dV/dt = k*(N/4-V); N')
K = np.zeros((3,3))*np.NaN
K[0,1] = K[1,0] = K[1,2] = K[2,1] = 1
print K
SparseConnection(G('V'), G('N'), K)

t, dt = 600.0, 0.1
for i in range(int(t/dt)):
    G.evaluate(dt=dt)
    G.V[0,:] = 0 
    G.V[:,n-1] = G.V[n-1,:] = G.V[:,0] = 1

fig = plt.figure(figsize=(10,7.5))
plt.imshow(G.V, cmap=plt.cm.hot, origin='lower', 
           interpolation='bicubic', vmin=0, vmax=1)
plt.colorbar()
CS = plt.contour(G.V, 10, colors='k')
plt.clabel(CS, inline=1, fontsize=16)
plt.grid(), plt.show()
