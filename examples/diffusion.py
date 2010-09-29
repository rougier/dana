#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Simple and naive model of heat diffusion
'''
from dana import *

n,k  = 40, .1
src = Group((n,n), '''dV/dt = k*N : float
                      N           : float''')
SparseConnection(src('V'), src('N'), np.array([[np.NaN, 1,  np.NaN], 
                                               [  1,    -4,   1],
                                               [np.NaN, 1,  np.NaN]]))
src.V = 1
for i in range(2500):
    src.run(dt=0.25)
    src.V[:,n-1] = src.V[n-1,:] = src.V[:,0] = 1
    src.V[0,:] = 0 
fig = plt.figure(figsize=(10,7.5))
plt.imshow(src.V, cmap=plt.cm.hot, origin='lower', extent=[0,n-1,0,n-1],
           interpolation='bicubic', vmin=0, vmax=1)
plt.colorbar()
CS = plt.contour(src.V, 10, colors='k')
plt.clabel(CS, inline=1, fontsize=16)
plt.grid(), plt.show()
