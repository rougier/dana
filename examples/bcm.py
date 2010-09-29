#!/usr/bin/env python
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
'''
Implementation of the BCM learning rule for learning selectivity to input. Ten
stimuli are randomly presented to the network. Each target unit learns to
become responsive to a single stimulus.  Final weights matrix should be tuned
such that each line represents a single stimulus (a single 1 on each line in
the displayed output).

Two target units may share the same stimulus selectivity and thus, some
stimuli may be not represented in the output.


References:
-----------

  E.L Bienenstock, L. Cooper, P. Munro, 'Theory for the development of neuron
  selectivity: orientation specificity and binocular interaction in visual
  cortex', The Journal of Neuroscience 2 (1): 32-48, 1982.

'''
from random import choice
from dana import *

n = 10
stims = np.identity(n)
src = Group((n,), ''' V = choice(stims)    : float ''')
tgt = Group((n,), ''' dC/dt = (F-C)*1.0    : float
                      dT/dt = (C**2-T)*0.1 : float
                      ----------------------------
                      F                    : float ''')
C = DenseConnection(src, tgt('F'), np.random.random((n,n)),
                    'dW/dt = pre.V*post.C*(post.C-post.T)*0.01')
run(n=10000)
for i in range(n):
    print 'Unit %d selective to ' % i, (C.weights[i] > 1e-3).astype(int)

