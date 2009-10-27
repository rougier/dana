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
import numpy, pylab, dana

# Simulation parameters
# ______________________________________________________________________________
n       = 100
dt      = 0.01
alpha   = 10.0
tau     = 0.10
h       = 0.0
    
# Build group and connect them
# ______________________________________________________________________________
I = dana.zeros(shape=(n,), name='I')
V = dana.zeros(shape=(n,), name='V')

# Connect them
# ______________________________________________________________________________
V.connect(I.V, numpy.ones((1,)), 'I', shared=True)
V.connect(V.V, 1.5*dana.gaussian(2*n+1,.1)-0.75*dana.gaussian(2*n+1,1.0),
               'L', shared=True)

# Set Dynamic Neural Field equation
# ______________________________________________________________________________
V.constant = {'n':n,
              'tau':tau,
              'h':h,
              'alpha':alpha,
              'minact':0.0}
V.dV = 'maximum(0,V+dt*(-V+(L*100.0/n+I+h)/alpha)/tau)'



# Run some iterations
# ______________________________________________________________________________
#I['V'] = 2.5*dana.gaussian(n, 0.1, -0.1) + 2.5*dana.gaussian(n, 0.1, +0.1)
#I['V'] = 2.5*dana.gaussian(n, 0.1, -0.25) + 2.5*dana.gaussian(n, 0.1, +0.25)
#I['V'] = 2.5*dana.gaussian(n, 0.1, -0.5) + 2.5*dana.gaussian(n, 0.1, +0.5)
I.V = 2.5*dana.gaussian(n, 0.1, -0.25) + 2.0*dana.gaussian(n, 0.1, +0.25)

V_hist = numpy.zeros((100,n))
for i in range(100):
    V.compute(dt)
    V_hist[i] = V.V

pylab.figure(figsize=(10,8))
pylab.axes([.1, .1, .8, .2])
pylab.xlabel('Espace',fontsize=16)
pylab.ylabel('Amplitude',fontsize=16)
X = numpy.arange(0, n, 1)
pylab.plot(X,I.V, linewidth=3, color=(.1,.1,1), linestyle='-', label='I(x)')
pylab.ylim(0,3)

pylab.axes([.1, .3, .8, .6])
pylab.pcolor(V_hist)
pylab.hot()
pylab.ylabel('Temps',fontsize=16)
pylab.xlabel('',fontsize=16)
pylab.yticks([])
pylab.xticks([])

ax=pylab.axes([.925, .1, .025, .8])
pylab.colorbar(cax=ax)

pylab.show()


