#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
from dana import *

n = 256
src = np.zeros((n,))
tgt = Group((n,), '''dU/dt = (-V + 0.1*L + I); V = maximum(U,0); I; L''')
SparseConnection(src, tgt('I'), np.ones((1,)))
SharedConnection(tgt('V'), tgt('L'), +1.00*gaussian(2*n+1, 0.10)
                                     -0.75*gaussian(2*n+1, 1.00))
src[...] = 1.1*gaussian(n, 0.1, -0.5) + 1.0*gaussian(n, 0.1, +0.5)
run(t=10.0, dt=0.1)

X = np.linspace(0.0, 1.0, n)
fig = plt.figure(figsize=(10,4))
fig.patch.set_alpha(0.0)
plt.plot(X, src,   linewidth=1, color='blue', label='I(x)')
plt.plot(X, tgt.V, linewidth=3, color='red',  label='U(x,t)')
plt.legend(['I(x)', 'U(x,t)'], 'upper right')
plt.show()
