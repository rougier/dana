#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import time
import numpy as np
from dana import *

class array(np.ndarray):
    pass

n = 50
t = 2.0
dt = 0.00001
epochs = int(t/dt) #100000

# Numpy regular array
Z = np.ones((n,n), dtype=np.double)
t0 = time.clock()
for i in range(epochs):
    Z += (Z+1)*dt
print 'numpy:                               ', time.clock()-t0

print
# Numpy regular array + dynamic evaluation
Z[...] = 0
t0 = time.clock()
for i in range(epochs):
    Z += eval("(Z+1)*dt")
print 'numpy + dynamic eval:                ', time.clock()-t0

# Numpy regular array + static evaluation
expr = compile("(Z+1)*dt", "<string>", "eval")
Z[...] = 0
t0 = time.clock()
for i in range(epochs):
    Z += eval(expr)
print 'numpy + static eval:                 ', time.clock()-t0

# Numpy regular array + dynamic execution
Z[...] = 0
t0 = time.clock()
for i in range(epochs):
    exec('Z += (Z+1)*dt')
print 'numpy + dynamic exec:                ', time.clock()-t0

# Numpy regular array + static execution
Z[...] = 0
expr = compile("Z += (Z+1)*dt", "<string>", "exec")
t0 = time.clock()
for i in range(epochs):
    exec(expr)
print 'numpy + static exec:                 ', time.clock()-t0

print
# Numpy aligned interleaved array
Z = np.zeros((n,n),dtype=[('x',np.double), ('y',np.int)])['x']
t0 = time.clock()
for i in range(epochs):
    Z += (Z+1)*dt
print 'aligned interleaved array:           ', time.clock()-t0

# Numpy unaligned interleaved array
Z = np.zeros((n,n),dtype=[('x',np.double), ('y',np.bool)])['x']
t0 = time.clock()
for i in range(epochs):
    Z += (Z+1)*dt
print 'unaligned interleaved array:         ', time.clock()-t0

print
# Numpy subclass array
z = np.zeros((n,n), dtype=np.double)
Z = z.view(np.ndarray)
t0 = time.clock()
for i in range(epochs):
    Z += (Z+1)*dt
print 'subclass array:                      ', time.clock()-t0

# Numpy aligned interleaved subclass array
z = np.zeros((n,n),dtype=[('x',np.double), ('y',np.int)])['x']
Z = z.view(array)
t0 = time.clock()
for i in range(epochs):
    Z += (Z+1)*dt
print 'aligned interleaved subclass array:  ', time.clock()-t0

# Numpy unaligned interleaved subclass array
z = np.zeros((n,n),dtype=[('x',np.double), ('y',np.bool)])['x']
Z = z.view(array)
t0 = time.clock()
for i in range(epochs):
    Z += (Z+1)*dt
print 'unaligned interleaved subclass array:', time.clock()-t0

print
eq = Equation('Z = (Z+1)*dt : double')
Z = np.zeros((n,n))
t0 = time.clock()
for i in range(epochs):
    Z += eq.evaluate(Z)
print 'dana equation:                       ', time.clock()-t0

eq = DifferentialEquation('dZ/dt = (Z+1) : double')
Z = np.zeros((n,n))
t0 = time.clock()
for i in range(epochs):
    eq.evaluate(Z,dt)
print 'dana differential equation:          ', time.clock()-t0

group = Group((n,n), 'dV/dt = (V+1) : double')
group.V = 0
t0 = time.clock()
group.run(t,dt)
print 'dana group:                          ', time.clock()-t0
