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
Simple and naive model of heat diffusion
'''
from dana import *

n,k  = 40, 4
src = Group((n,n), '''dV/dt = k*N : float
                      N           : float''')
SparseConnection(src('V'), src('N'), np.array([[np.NaN, 1,  np.NaN], 
                                               [  1,    -4,   1],
                                               [np.NaN, 1,  np.NaN]]))
src.V = 1

@after(clock.tick)
def set_border(time):
    src['V'][:,n-1] = src.V[n-1,:] = src.V[:,0] = 1
    src['V'][0,:] = 0 

run(time=10.0*second, dt=5*millisecond)

fig = plt.figure(figsize=(10,7.5))
plt.imshow(src.V, cmap=plt.cm.hot, origin='lower', extent=[0,n-1,0,n-1],
           interpolation='bicubic', vmin=0, vmax=1)
plt.colorbar()
CS = plt.contour(src.V, 10, colors='k')
plt.clabel(CS, inline=1, fontsize=16)
plt.grid(), plt.show()
