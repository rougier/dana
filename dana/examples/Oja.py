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
Implementation of the Oja learning rule for extracting the principal component of an elliptical gaussian distribution

References:
-----------
  E. Oja. "A Simplified Neuron Model as a Principal Component Analyzer"
  Journal of Mathematical Biology 15: 267-273, 1982 
'''

import time
import numpy, dana

src = dana.zeros((2,))
oja = dana.zeros((1,), keys=['V'])
K = numpy.random.random((oja.size, src.size))
oja.connect(src, K, 'F')

eta = 0.001
oja.dV = '-V + F'
oja.dF = 'eta*post.V*(pre.V-post.V*post.F._kernel)'

''' Function for generating a random sample according to an elliptical gaussian distribution'''
def getSample(theta, mu1, std1, mu2, std2):
    u1 = numpy.random.random();
    u2 = numpy.random.random();
    T1 = numpy.sqrt(-2 * numpy.log(u1)) * numpy.cos(2*numpy.pi*u2);
    T2 = numpy.sqrt(-2 * numpy.log(u1)) * numpy.sin(2*numpy.pi*u2);
    x = mu1 +  (std1*T1 * numpy.cos(theta) - std2*T2 * numpy.sin(theta));
    y = mu2 +  (std1*T1 * numpy.sin(theta) + std2*T2 * numpy.cos(theta));	
    return [x,y]


t = time.clock()
theta = -135.0 * numpy.pi / 180.0
for i in xrange(10000):
    [x,y] = getSample(theta, 0.0, 1.0, 0.0, 0.5)
    src.V = [x,y]
    oja.compute()
    oja.learn()
print "Execution time ",time.clock()-t," s."

print "Learned weights : ", oja.F._kernel
# Given the distribution is elliptical, its Principal Component should be oriented
# along the main axis of the distribution. Therefore the weights should be +/- (cos(theta),sin(theta))
print "It should have converged to +/- (", numpy.cos(theta), ",", numpy.sin(theta),")"
