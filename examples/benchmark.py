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
import numpy as np
from dana import *

class array(np.ndarray):
    pass

n = 100
t = 5.0
dt = 0.0001
epochs = int(t/dt)

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
    Z += eq.evaluate(dt,Z)
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
for i in range(epochs):
    group.evaluate(dt=dt)
print 'dana group:                          ', time.clock()-t0

group.V = 0
t0 = time.clock()
eq = group._model._diff_equations[0]
for i in range(epochs):
    eq.evaluate(group['V'],dt)
print 'dana group 2:                        ', time.clock()-t0
