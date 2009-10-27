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
source = dana.group((N,1), name='source')
bcm    = dana.group((N,1), fields=['c','t'], name='bcm')

K = numpy.random.random(bcm.shape + source.shape)
bcm.connect(source['V'],'F',K,shared=False)
    

stims = numpy.identity(N)


TAU     = 1.0
TAU_BAR = TAU * 0.1
ETA     = TAU_BAR * 0.1


# Set BCM equations
# ______________________________________________________________________________
bcm.constant['tau']     = TAU
bcm.constant['tau_bar'] = TAU_BAR
bcm.constant['eta']     = ETA
bcm.equation['c']       = "c + (F - c) * tau"
bcm.equation['t']       = "t + (c**2 - t) * tau_bar"
bcm.equation['F']       = "W + pre['V'] * post['c'] * (post['c'] - post['t']) * eta"
bcm['c']                = 1
bcm['t']                = 1

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
view = dana.pylab.view([source['V'], bcm['c']])
view.show()
