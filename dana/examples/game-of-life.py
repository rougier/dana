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
G = dana.group((n,n), dtype=int)

# Connections
# ______________________________________________________________________________
K = numpy.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]])
G.connect(G.V, K, 'N', sparse=True)



# Set Dynamic Neural Field equation
# ______________________________________________________________________________
G.dV = '-V+maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'

# Set input
# ______________________________________________________________________________
G.V = numpy.random.randint(0,2,G.shape)

# Run some iterations
# ______________________________________________________________________________
S = numpy.zeros(G.shape)
for i in range(50):
    G.compute()
    S += G.V
S /= 50.0

# Display result using pylab
# __________________________________________________________________________
pylab.imshow(numpy.maximum(S,G.V), cmap=pylab.cm.hot, interpolation='nearest')
pylab.colorbar()
pylab.show()
#view = dana.pylab.view(numpy.maximum(S,G.V), cmap=pylab.cm.hot, vmin=0, vmax=1)
#pylab.xticks([])
#pylab.yticks([])
#view.show()

