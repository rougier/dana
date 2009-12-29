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
#  If you modify  this software, you should include a notice  giving the name of
#  the person  performing the  modification, the date  of modification,  and the
#  reason for such modification.
# 
#  Contact: 
# 
#      CORTEX Project - INRIA
#      INRIA Lorraine, 
#      Campus Scientifique, BP 239
#      54506 VANDOEUVRE-LES-NANCY CEDEX 
#      FRANCE
# 
import numpy, pylab
import dana, dana.pylab

# Build groups
# ______________________________________________________________________________
n = 100
G = dana.group((n,n))

# Connections
# ______________________________________________________________________________
K = numpy.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]])
G.connect(G.V, K, 'N', shared=True)

# Set Dynamic Neural Field equation
# ______________________________________________________________________________
k = 1
G.dV = 'V+dt*k*(N/8-V)'

# Set input
# ______________________________________________________________________________
G.V = -numpy.random.ranf((n,n))

# Run some iterations
# ______________________________________________________________________________
for i in range(200):
    G.V[0,:] = G.V[n-1,:] = G.V[:,0] = 1
    G.V[:,n-1] = -1
    G.compute(1)
G.V[0,:] = G.V[n-1,:] = G.V[:,0] = 1
G.V[:,n-1] = -1

# Display result using pylab
# __________________________________________________________________________
view = dana.pylab.view(G.V, vmin=-1, vmax=1,
                       cmap=pylab.cm.hot, interpolation='bicubic')
view.show()
