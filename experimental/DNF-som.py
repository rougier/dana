#!/usr/bin/env python
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
'''
'''

from dana import *

n       = 100
dt      = 0.1
alpha   = 10.0
tau     = 1.0
h       = 0.0
epsilon = 0.05
lrate   = 0.125

class DistanceConnection(Connection):
    def __init__(self, source=None, target=None, weights=None, equation = ''):
        Connection.__init__(self, source, target)
        self._weights = weights
        self.setup_equation(equation)
    def output(self):
        R = np.abs(self._weights - self._actual_source)
        return R.reshape(self.target.shape)

stimulus = np.ones((1,))
som = Group((n,), ''' dU/dt = -V+(Le-Li+1-I)/alpha : float
                       V    = maximum(U,0)         : float
                       I                           : float
                       Le                          : float
                       Li                          : float ''')
DenseConnection(som('V'), som('Le'), 1.50*gaussian(2*n+1, 0.1)) 
DenseConnection(som('V'), som('Li'), 0.75*gaussian(2*n+1, 1.0))
I = DistanceConnection(stimulus, som('I'), rnd.rand(n,1),
                       'dW/dt = lrate*(Le/n)*(pre-W)')
som.setup()
for i in range(2500):
    if i%100 == 0: print i
    som.V = som.U = 0
    stimulus[...] = rnd.randint(3)/2.0
    dV = 1
    while dV > epsilon:
        V = som.V.copy()
        som.run(dt)
        dV = abs(som.V-V).sum()

plt.figure(figsize=(10,6))
plt.plot(I.weights, linewidth=3, color='red')
plt.show()
