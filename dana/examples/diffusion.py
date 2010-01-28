#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE
'''

Numerical simulation of the heat equation of the form:

∂u    ⎛ ∂²u   ∂²u ⎞                      ∂u
-- -α ⎜ --- + --- ⎟ = 0 or equivalently  -- = α∇²u
∂t    ⎝ ∂x²   ∂y² ⎠                      ∂t

For a function u(x,y,t) of two spatial variables (x,y) and the time variable t.

'''
import numpy, dana
import matplotlib.pyplot as plt

n = 60
k = 1
G = dana.group((n,n))
K = numpy.array([[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]])
G.connect(G.V, K, 'N', shared=True)
G.dV = 'dt*k*(N/4-V)'

for i in range(2500):
    G.compute(dt=1)
    G.V[0,:] = 0 
    G.V[:,n-1] = G.V[n-1,:] = G.V[:,0] = 1

fig = plt.figure(figsize=(10,7.5))
plt.imshow(G.V, cmap=plt.cm.hot, origin='lower',
           extent=[0,n,0,n],
           interpolation='bicubic', vmin=0, vmax=1)
plt.colorbar()
CS = plt.contour(G.V, 5, colors='k')
plt.clabel(CS, inline=1, fontsize=16)
plt.grid()
plt.show()
