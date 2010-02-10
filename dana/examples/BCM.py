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
Implementation of the BCM learning rule.

References:
-----------
  E.L Bienenstock, L. Cooper, P. Munro. "Theory for the development of neuron
  selectivity: orientation specificity and binocular interaction in visual
  cortex". The Journal of Neuroscience 2 (1): 32—48, 1982.
'''
import time
import numpy, dana
from random import choice

n = 10
src = dana.ones((n,))
bcm = dana.ones((n,), keys=['C','T'])
K = numpy.random.random((bcm.size, src.size))
bcm.connect(src, K, 'F')
stims = numpy.identity(n)
tau = 1.0
tau_ = tau * 0.1
eta = tau_ * 0.1

bcm.dC = '(F-C)*1.0'
bcm.dT = '(C**2-T)*0.1'
bcm.dF = 'pre.V*post.C*(post.C-post.T)*0.01'

t = time.clock()
V = src.V
for i in xrange(10000):
    V[:] = choice(stims).reshape(src.shape)
    bcm.compute()
    bcm.learn()
print time.clock()-t

print 'Learned prototypes'
for i in range(n):
    print 'Unit %d: ' % i, (bcm.F._kernel[i] > 1e-3).astype(int)
