#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  ____  _____ _____ _____ 
# |    \|  _  |   | |  _  |   DANA, Distributed Asynchronous Adaptive Numerical
# |  |  |     | | | |     |         Computing Framework
# |____/|__|__|_|___|__|__|         Copyright (C) 2009 INRIA  -  CORTEX Project
#                        
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either  version 3 of the  License, or (at your  option) any later
# version.
#
# This program is  distributed in the hope that it will  be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public  License for  more
# details.
#
# You should have received a copy  of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#
#  Contact: 
#
#     CORTEX Project - INRIA
#     INRIA Lorraine, 
#     Campus Scientifique, BP 239
#     54506 VANDOEUVRE-LES-NANCY CEDEX 
#     FRANCE
#
import pylab, numpy, dana

# Simulation parameters
# ______________________________________________________________________________
n       = 100
dt      = 0.1
alpha   = 10.0
tau     = 1.0
h       = 0.0
epsilon = 0.05
lrate   = 0.125

# Build group and connect them
# ______________________________________________________________________________
input = dana.zeros(shape=(1,), name='I')
som = dana.zeros(shape=(n,), name='V')

# Connect them
# ______________________________________________________________________________
som.connect(input.V, numpy.random.rand(n,1), 'I-', shared=False, sparse=True)
som.connect(som.V, 1.50*dana.gaussian(2*n+1, 0.1), 'Le*', shared=False)
som.connect(som.V, 0.75*dana.gaussian(2*n+1, 1.0), 'Li*', shared=False)

# Set Dynamic Neural Field equation
# ______________________________________________________________________________
som.dV = "-V+maximum(0, V+dt*(-V+((Le-Li)*100.0/n+(1-I)+h)/alpha)/tau)"
som.dI = "lrate*Le/n*(pre.V-W)"

# Run some iterations
# ______________________________________________________________________________
for i in range(2500):
    if (i%100) == 0: print i
    som.V = 0
    input.V = numpy.random.randint(3)/2.0
    dV = 1
    while dV > epsilon:
        V = som.V.copy()
        som.compute(dt)
        dV = abs(som.V-V).sum()
    som.learn(dt)

# Display result using pylab
# ______________________________________________________________________________
pylab.figure(figsize=(10,6))
pylab.plot(som.I.kernel, linewidth=3, color=(1,.1,.1), linestyle='-', label='W')
#pylab.plot(som.V, linewidth=3, color=(.1,.1,1), linestyle='-', label='som(x,t)')
pylab.axis([0,n, -0.1, 1.1])
pylab.show()

