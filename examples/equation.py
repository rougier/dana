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
