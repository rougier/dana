w#!/usr/bin/env python
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


n = 50
k = 2.5
G = zeros((n,n), 'dV/dt = k*(N/4-V); N')
K = np.zeros((3,3))*np.NaN
K[0,1] = K[1,0] = K[1,2] = K[2,1] = 1
print K
SparseConnection(G('V'), G('N'), K)

t, dt = 600.0, 0.1
for i in range(int(t/dt)):
    G.evaluate(dt=dt)
    G.V[0,:] = 0 
    G.V[:,n-1] = G.V[n-1,:] = G.V[:,0] = 1

fig = plt.figure(figsize=(10,7.5))
plt.imshow(G.V, cmap=plt.cm.hot, origin='lower', 
           interpolation='bicubic', vmin=0, vmax=1)
plt.colorbar()
CS = plt.contour(G.V, 10, colors='k')
plt.clabel(CS, inline=1, fontsize=16)
plt.grid(), plt.show()
