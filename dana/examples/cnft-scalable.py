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
import time
import dana, numpy
import dana.pylab


N       = 60
dt      = 0.1
alpha   = 10.0
tau     = 1.0
h       = 0.0
input = dana.zeros((N,N), name='input')
focus = dana.zeros((N,N), name='focus')

Wi = numpy.ones((1,1))
focus.connect(input.V, Wi, 'I', sparse=True)
Wf = 1.25*dana.gaussian((2*N+1,2*N+1), 0.1) \
    -0.70*dana.gaussian((2*N+1,2*N+1), 1)

focus.connect (focus.V, Wf, 'L', shared=True)
focus.dV = '-V+maximum(V+dt/tau*(-V+(L/(N*N)*40*40+I+h)/alpha),0)'

input.V  = dana.gaussian((N,N), 0.2, ( 0.5, 0.5))
input.V += dana.gaussian((N,N), 0.2, (-0.5,-0.5))
input.V += (2*numpy.random.random((N,N))-1)*.05

n = 500
t =time.clock()
for i in range(n):
    focus.compute(dt)
print time.clock()-t
dana.pylab.view([input.V, focus.V]).show()
