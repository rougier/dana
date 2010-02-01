#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009, 2010 Nicolas Rougier - INRIA - CORTEX Project
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
Numerical integration of dynamic neural fields
----------------------------------------------
This script implements the numerical integration of dynamic neural fields [1]_
of the form:
                  
1 ∂U(x,t)             1 ⌠+∞                        1 
- ------- = -U(x,t) + - ⎮   W(|x-y|)f(U(y,t)) dy + - I(x,t)
α   ∂t                τ ⌡-∞                        τ

where # U(x,t) is the potential of a neural population at position x and time t.
      # W(x) is a neighborhood function
      # f(x) is the firing rate of a single neuron.
      # α is the temporal decay of the synapse.
      # τ is a scaling term
      # I(x,t) is the input at position x.

References
----------
    _[1] http://www.scholarpedia.org/article/Neural_fields
'''
import numpy, dana
import matplotlib.pyplot as plt

n       = 100
dt      = 0.05
alpha   = 10.0
tau     = 1.0
h       = 0.0
    
I = dana.zeros(shape=(n,), name='I')
V = dana.zeros(shape=(n,), name='V')
V.connect(I.V, numpy.ones((1,)),'I', shared=True)
V.connect(V.V, 1.50*dana.gaussian(2*n+1,.1)
              -0.75*dana.gaussian(2*n+1,1.0),
          'L', shared=True)
V.dV = '-V+maximum(0,V+dt*(-V+(L*100.0/n+I+h)/alpha)/tau)'

I['V'] = .5
for i in range(2500):
    V.compute(dt)

X = numpy.arange(0.0, 1.0, 1.0/n)
plt.figure(figsize=(10,6))
plt.plot(X,I['V'], linewidth=3, color=(.1,.1,1), linestyle='-', label='I(x)')
plt.plot(X,V['V'], linewidth=3, color=(1,.1,.1), linestyle='-', label='V(x,t)')
plt.axis([0,1, -0.1, 1.1])
plt.show()

