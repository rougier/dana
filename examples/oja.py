#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Implementation of the Oja learning rule for extracting the principal component
of an elliptical gaussian distribution. Given that the distribution is
elliptical, its principal component should be oriented along the main axis of
the distribution, therefore, final weights should be +/-cos(theta), sin(theta)


References:
-----------

  E. Oja, "A Simplified Neuron Model as a Principal Component Analyzer"
  Journal of Mathematical Biology 15: 267-273, 1982.

'''
from numpy import *
from dana import *

def sample(theta, mu1, std1, mu2, std2):
    ''' Random sample according to an elliptical Gaussian distribution'''
    u1 = random.random()
    u2 = random.random()
    T1 = sqrt(-2.0*log(u1))*cos(2.0*pi*u2)
    T2 = sqrt(-2.0*log(u1))*sin(2.0*pi*u2)
    x = mu1 + (std1*T1*cos(theta) - std2*T2*sin(theta))
    y = mu2 + (std1*T1*sin(theta) + std2*T2*cos(theta))	
    return x,y

theta = -135.0 * pi / 180.0
src = Group((2,), 'V = sample(theta,0.0,1.0,0.0,0.5)')
tgt = Group((1,), 'V')
C = DenseConnection(src, tgt, np.ones((1,2)),
                    'dW/dt = post*(pre-post*W)*0.001')
run(n=10000)
print "Learned weights : ", C.weights[0]
print "(should be +/- [%f, %f])" % (cos(theta), sin(theta))
