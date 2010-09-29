#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import time
import numpy
from dana import *

eq = DifferentialEquation('dv/dt = a+(b)*v : float')
n = 10000
t  = 1.0
dt = 0.00005
v = numpy.ones((n,))
a = numpy.random.random((n,))
b = 1

def integrate(method):
    v[...] = 1
    t0 = time.clock()    
    eq.select(method)
    for i in range(1,int(t/dt)+1):
        eq.evaluate(v,dt,a,b)
    print '  time', time.clock()-t0
    print '  value', v[0]
    print

print 'Numpy Direct Euler integration'
v[...] = 1
t0 = time.clock()
for i in range(1,int(t/dt)+1):
    v += a*dt + b*dt*v
print '  time', time.clock()-t0
print '  value', v[0]
print

print 'Forward Euler integration'
integrate('Forward Euler')

print 'Runge Kutta second order integration'
integrate('Runge Kutta 2')

print 'Runge Kutta fourth order integration'
integrate('Runge Kutta 4')

print 'Exponential Euler integration'
integrate('Exponential Euler')
