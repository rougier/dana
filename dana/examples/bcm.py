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


import numpy, dana, time, dana.pylab
from random import choice

N = 10
source = dana.ones((N,1), name='source')
bcm    = dana.ones((N,1), keys=['C','T'], name='bcm')

K = numpy.random.random((bcm.size,source.size))
bcm.connect(source.V, K, 'F',shared=False)
    

stims = numpy.identity(N)


tau = 1.0
tau_bar = tau * 0.1
eta = tau_bar * 0.1


# Set BCM equations
# ______________________________________________________________________________
bcm.dC = "C + (F - C)*tau"
bcm.dT = "T + (C**2 - T) * tau_bar"
bcm.dF = "W + pre['V'] * post['C'] * (post['C'] - post['T']) * eta"

# Run some iterations
# ______________________________________________________________________________
n = 10000
t = time.clock()
for i in range(n):
    source['V'] = choice(stims).reshape(source.shape)
    bcm.compute()
    bcm.learn()
print time.clock()-t

# Display result using pylab
# __________________________________________________________________________
view = dana.pylab.view([source.V, bcm.C])
view.show()
