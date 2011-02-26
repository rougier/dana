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
SigmaPiConnection
'''
from dana import *

class SigmaPiConnection(Connection):
    def __init__(self, source=None, modulator=None, target=None, scale=1.0):
        Connection.__init__(self, source, target)
        self._scale = scale
        # Get actual modulator
        names = modulator.dtype.names
        if names == None:
            self._actual_modulator = modulator
        else:
            self._actual_modulator = (modulator[names[0]])
    def output(self):
        src = self._actual_source
        mod = self._actual_modulator
        tgt = self._actual_target
        if len(tgt.shape) == len(src.shape) == len(mod.shape) == 1:
            return convolve1d(src,mod[::1])*self._scale
        elif len(tgt.shape) == len(src.shape) == len(mod.shape) == 2:
            return convolve2d(src,mod[::-1,::-1])*self._scale
        else:
            raise NotImplemented

# 1 dimension
# -----------
n = 100
src = 1.00*gaussian((n,), 10.0/float(n), +0.5) \
    + 1.00*gaussian((n,),  5.0/float(n), -0.5) \
    + 0.05*rnd.random((n,))
cmd = gaussian((n,), 3.0/float(n), 0.25)
tgt = np.zeros((n,))
SigmaPiConnection(src,cmd,tgt,scale=0.1).propagate()
plt.subplot(3,1,1), plt.plot(src), plt.title('Input')
plt.subplot(3,1,2), plt.plot(cmd), plt.title('Command')
plt.subplot(3,1,3), plt.plot(tgt), plt.title('Output')
plt.show()

# 2 dimensions
# ------------
n = 50
src = 1.00*gaussian((n,n), 10.0/float(n), (+0.5,+0.5)) \
    + 0.50*gaussian((n,n),  5.0/float(n), (-0.5,-0.5)) \
    + 0.05*rnd.random((n,n))
cmd = gaussian((n,n), 5.0/float(n), (0.5,0.25))
tgt = np.zeros((n,n))
SigmaPiConnection(src,cmd,tgt,scale=0.1).propagate()

plt.figure(figsize=(18,6))
plt.subplot(1,3,1), plt.imshow(src), plt.title('Input')
plt.subplot(1,3,2), plt.imshow(cmd), plt.title('Command')
plt.subplot(1,3,3), plt.imshow(tgt), plt.title('Output')
plt.show()
