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
import dana
import numpy as np


# Simulation parameters
# ______________________________________________________________________________
n       = 40
dt      = 0.1
alpha   = 12.5
tau     = 0.75
h       = 0.1
min_act = -1.0
max_act =  1.0

# Build groups
# ______________________________________________________________________________
input = dana.group((n,n), name='input')
focus = dana.group((n,n), name='focus')

# Connections
# ______________________________________________________________________________
W = np.ones((1,1))*1.5
focus.connect(input.V, W, 'I', shared=True)
W = 3.15*dana.gaussian((2*n+1,2*n+1), 0.05) - 0.7*dana.gaussian((2*n+1,2*n+1), 0.1)
W[n,n] = 0
focus.connect (focus.V, W, 'L', shared=True)


# Set Dynamic Neural Field equation
# ______________________________________________________________________________
focus.dV = 'minimum(maximum(V+dt/tau*(-V+(L+I+h)/alpha),min_act), max_act)'

# Set input
# ______________________________________________________________________________
input.V  = dana.gaussian((n,n), 0.2, ( 0.5, 0.5))
input.V += dana.gaussian((n,n), 0.2, (-0.5,-0.5))
input.V += (2*numpy.random.random((n,n))-1)*.05

# Run some iterations
# ______________________________________________________________________________
n = 250
t = time.clock()
for i in range(n):
    focus.compute(dt)
print time.clock()-t
