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

In the following example, two stimuli are presented to the DNF and the DNF
stabilizes itself onto one of the two stimuli because of the presence of noise.
If one removes noise, two small bumps of activity will exist within the focus
group.

:References:
    _[1] http://www.scholarpedia.org/article/Neural_fields
'''
from dana import *

n = 40
p = 2*n+1
alpha, tau, h = 1.0, 0.1, 0

input = np.zeros((n,n))
focus = Group((n,n), '''dU/dt = alpha*(-V + tau*(L+I)) +h : float
                         V    = np.maximum(U,0)           : float
                         I                                : float
                         L                                : float''')
SparseConnection(input, focus('I'), np.ones((1,1)))
SharedConnection(focus('V'), focus('L'),
                 1.25*gaussian((p,p),0.1) - 0.75*gaussian((p,p),1.0))
input[...] = gaussian((n,n),0.25,(0.5,0.5))   \
           + gaussian((n,n),0.25,(-0.5,-0.5)) \
           + (2*rnd.random((n,n))-1)*.05
run(t=5.0, dt=0.01)



fig = plt.figure(figsize=(12,5))
plt.subplot(121)
plt.imshow(input, origin='lower', cmap = plt.cm.Purples,
           interpolation='nearest', extent=[0,n,0,n])
plt.text(1,1, "Input", fontsize=24)
plt.yticks(np.arange(focus.shape[0]//10+1)*10)
plt.xticks(np.arange(focus.shape[1]//10+1)*10)
plt.grid()
plt.subplot(122)
plt.imshow(focus.V, origin='lower', cmap = plt.cm.Purples,
           interpolation='nearest', extent=[0,n,0,n])
plt.text(1,1, "Focus", fontsize=24)
plt.yticks(np.arange(focus.shape[0]//10+1)*10)
plt.xticks(np.arange(focus.shape[1]//10+1)*10)
plt.grid()
plt.show()
