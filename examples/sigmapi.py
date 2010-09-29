#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
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
