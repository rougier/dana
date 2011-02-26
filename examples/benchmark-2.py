#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
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
