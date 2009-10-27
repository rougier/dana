#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#   ____  _____ _____ _____ 
#  |    \|  _  |   | |  _  |   DANA, Distributed Asynchronous Adaptive Numerical
#  |  |  |     | | | |     |         Computing Framework
#  |____/|__|__|_|___|__|__|         Copyright (C) 2009 INRIA  -  CORTEX Project
#                         
#  This program is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free Software
#  Foundation, either  version 3 of the  License, or (at your  option) any later
#  version.
# 
#  This program is  distributed in the hope that it will  be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public  License for  more
#  details.
# 
#  You should have received a copy  of the GNU General Public License along with
#  this program. If not, see <http://www.gnu.org/licenses/>.
# 
#  Contact: 
# 
#      CORTEX Project - INRIA
#      INRIA Lorraine, 
#      Campus Scientifique, BP 239
#      54506 VANDOEUVRE-LES-NANCY CEDEX 
#      FRANCE
# 
import time, numpy
import dana, dana.pylab


# Simulation parameters
# ______________________________________________________________________________
N       = 60
dt      = 0.1
alpha   = 10.0
tau     = 1.0
h       = 0.0

# Build groups
# ______________________________________________________________________________
input = dana.group((N,N), name='input')
focus = dana.group((N,N), name='focus')

# Connections
# ______________________________________________________________________________
Wi = numpy.ones((1,1))
focus.connect(input.V, Wi, 'I', shared=True)
Wf = 1.25*dana.gaussian((2*N+1,2*N+1), 0.1) - 0.7*dana.gaussian((2*N+1,2*N+1), 1)
focus.connect (focus.V, Wf, 'L', shared=True)

# Set Dynamic Neural Field equation
# ______________________________________________________________________________
focus.dV = 'maximum(V+dt/tau*(-V+(L/(N*N)*40*40+I+h)/alpha),0)'
#focus.dV = 'maximum(V+dt/tau*(-V+(L+I+h)/alpha),0)'


# Set input
# ______________________________________________________________________________
input.V  = dana.gaussian((N,N), 0.2, ( 0.5, 0.5))
input.V += dana.gaussian((N,N), 0.2, (-0.5,-0.5))
input.V += (2*numpy.random.random((N,N))-1)*.05

# Run some iterations
# ______________________________________________________________________________
n = 500
t = time.clock()
for i in range(n):
    focus.compute(dt)
print time.clock()-t

# Display result using pylab
# __________________________________________________________________________
view = dana.pylab.view([input.V, focus.V])
view.show()
