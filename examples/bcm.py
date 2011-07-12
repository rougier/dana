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
tau_c = 1.0
tau_t = tau_c * 0.1
eta   = tau_t * 0.1
stims = np.identity(n)
src = np.zeros(n)
tgt = ones(n, ''' C = C + (F-C)*tau_c    : float
                  T = T + (C**2-T)*tau_t : float
                  ----------------------------
                  F                    : float ''')
C = DenseConnection(src, tgt('F'), np.random.random((n,n)),
                    'dW/dt = pre*post.C*(post.C-post.T)*eta')

@before(clock.tick)
def set_stimulus(*args):
    src[:] = choice(stims)
run(n=10000)
for i in range(n):
    print 'Unit %d selective to ' % i, (C.weights[i] > 1e-2).astype(int)
