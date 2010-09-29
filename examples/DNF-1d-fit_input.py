#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Numerical integration of dynamic neural fields
----------------------------------------------
This script implements the numerical integration of dynamic neural fields [1]_
of the form:
 
α ∂U(x,t)/∂t = -U(x,t) + τ*(∫ w(|x-y|).f(U(y,t)).dy + I(x,t) + h )

where U(x,t) is the potential of a neural population at position x and time t
      W(d) is a neighborhood function from ℝ⁺ → ℝ
      f(u) is the firing rate of a single neuron from ℝ → ℝ
      I(x,t) is the input at position x and time t
      h is the resting potential
      α is the temporal decay of the synapse
      τ is a scaling term

In the following example, A constant input is presented to the DNF and the DNF
stabilizes itself such that the peak of the bump of activity is at the input
level.

:References:
    _[1] http://www.scholarpedia.org/article/Neural_fields
'''
from dana import *

n = 100
src = np.zeros((n,))
tgt = Group((n,), '''dU/dt = (-V + 0.1*(L+I)) : float
                      V    = np.maximum(U,0)  : float
                      I                       : float
                      L                       : float''')
SparseConnection(src, tgt('I'), np.ones((1,)))
weights = 100.0/n*((1.50*gaussian(2*n+1,0.1) - 0.75*gaussian(2*n+1,1.0)))
SharedConnection(tgt('V'), tgt('L'), weights)

src[...] = .45
run(t=60.0, dt=0.1)

X = np.arange(0.0, 1.0, 1.0/n)
plt.figure(figsize=(10,6))
plt.plot(X, src,   linewidth=3, color='blue', label='I(x)')
plt.plot(X, tgt.V, linewidth=3, color='red',  label='V(x,t)')
plt.axis([0,1, -0.1, 1.1])
plt.show()
