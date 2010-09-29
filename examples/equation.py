#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Illustration of the different integration methods for differential equations.

References:
-----------

  http://en.wikipedia.org/wiki/Numerical_ordinary_differential_equations
'''
from dana import *

def integrate(method):
    eq.select(method)
    y, n = 1, int(t/dt)
    Y = np.zeros(n+1)
    Y[0] = y
    for i in range(1,n+1):
        y = eq.evaluate(y,dt)
        Y[i] = y
    return Y

eq = DifferentialEquation('dy/dt = 0+(1)*y : float')
t, dt  = 10.0, 2.0
X = np.linspace(0, t, t/dt+1)
Y_euler = integrate('Forward Euler')
Y_rk2   = integrate('Runge Kutta 2')
Y_rk4   = integrate('Runge Kutta 4')
Y_exp   = integrate('Exponential Euler')
plt.plot(X, Y_euler,    lw=2)
plt.plot(X, Y_rk2,      lw=2)
plt.plot(X, Y_rk4,      lw=2)
plt.plot(X, Y_exp,      lw=2)
X = np.linspace(0, t, 100)
Y_exact = np.exp(X)

plt.plot(X, Y_exact, lw=3)
plt.legend(['Forward Euler', 'Runge Kutta 2',
            'Runge Kutta 4', 'Exponential Euler',
            'Exact solution',], 'upper left')
plt.show()
