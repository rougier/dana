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

<<<<<<< .mine
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

=======
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
dt      = 0.01
alpha   = 10.0
tau     = 0.10
h       = 0.0
I = dana.zeros(shape=(n,), name='I')
V = dana.zeros(shape=(n,), keys=['U','V'], name='V')
V.connect((I,'V'), numpy.ones((1,)), 'I', shared=True)
V.connect((V,'U'),  3.00*dana.gaussian(2*n+1, 0.05)
                   -0.75*dana.gaussian(2*n+1, 0.20),
                   'L', shared=True)

V.dV = 'dt*(-V+(L*100.0/n+I+h)/alpha)/tau'
V.dU = '1.0/(1.0+exp(2.0-3.0*V))-U'

#I['V'] = 2.5*dana.gaussian(n, 0.1, -0.1) + 2.5*dana.gaussian(n, 0.1, +0.1)
#I['V'] = 2.5*dana.gaussian(n, 0.1, -0.25) + 2.5*dana.gaussian(n, 0.1, +0.25)
#I['V'] = 2.5*dana.gaussian(n, 0.1, -0.5) + 2.5*dana.gaussian(n, 0.1, +0.5)
I.V = 2.5*dana.gaussian(n, 0.1, -0.25) + 2.0*dana.gaussian(n, 0.1, +0.25)

V_hist = numpy.zeros((100,n))
for i in range(100):
    V.compute(dt)
    V_hist[i] = V.U
plt.figure(figsize=(10,8))
plt.axes([.1, .1, .8, .2])
plt.xlabel('Espace',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
X = numpy.arange(0, n, 1)
plt.plot(X,I.V, linewidth=3, color=(.1,.1,1), linestyle='-', label='I(x)')
plt.ylim(0,3)

plt.axes([.1, .3, .8, .6])
plt.pcolor(V_hist)
plt.hot()
plt.ylabel('Temps',fontsize=16)
plt.xlabel('',fontsize=16)
plt.yticks([])
plt.xticks([])

ax=plt.axes([.925, .1, .025, .8])
plt.colorbar(cax=ax)

plt.show()
