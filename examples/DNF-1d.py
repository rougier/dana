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
from dana import *

n = 256
src = np.zeros((n,))
tgt = Group((n,), '''dU/dt = (-V + 0.1*L + I); V = np.maximum(U,0); I; L''')
SparseConnection(src, tgt('I'), np.ones((1,)))
SharedConnection(tgt('V'), tgt('L'), +1.00*gaussian(2*n+1, 0.10)
                                     -0.75*gaussian(2*n+1, 1.00))
src[...] = 1.1*gaussian(n, 0.1, -0.5) + 1.0*gaussian(n, 0.1, +0.5)
run(time=10.0, dt=0.1)

X = np.linspace(0.0, 1.0, n)
fig = plt.figure(figsize=(10,4))
fig.patch.set_alpha(0.0)
plt.plot(X, src,   linewidth=1, color='blue', label='I(x)')
plt.plot(X, tgt.V, linewidth=3, color='red',  label='U(x,t)')
plt.legend(['I(x)', 'U(x,t)'], 'upper right')
plt.show()
