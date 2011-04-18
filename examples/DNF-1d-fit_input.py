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
run(time=60.0, dt=0.1)

X = np.arange(0.0, 1.0, 1.0/n)
plt.figure(figsize=(10,6))
plt.plot(X, src,   linewidth=3, color='blue', label='I(x)')
plt.plot(X, tgt.V, linewidth=3, color='red',  label='V(x,t)')
plt.axis([0,1, -0.1, 1.1])
plt.show()
