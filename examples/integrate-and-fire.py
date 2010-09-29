#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Implementation of integrate and fire spiking neuron model.

This is only a proof of concept. If you really need to work with spikes, you'd
better use brian simulator which is dedicated to spiking neural networks:

  http://www.briansimulator.org


References:
-----------

  L.F. Abbott, "Lapique's introduction of the integrate-and-fire model neuron
  (1907)" Brain Research Bulletin 50 (5/6): 303-3, 1999.

'''
from dana import *

ms  =  0.001   # millisecond
mV  =  0.001   # millivolt
tau = 20.0*ms  # membrane time constant
Vt  =-50.0*mV  # spike threshold
Vr  =-60.0*mV  # reset value
El  =-49.0*mV  # resting potential
psp =  0.5*mV  # postsynaptic potential size
sparseness = 0.1
n = 40

# Setup
# -----
src = Group((n,), '''dV/dt = -(V-El)/tau + I*psp/dt : float
                      S    = V > Vt                 : float
                      I                             : float''')
W = rnd.uniform(0,1,(n,n)) * (rnd.random((n,n)) < sparseness)
C = DenseConnection(src('S'), src('I'), W)
src.V = Vr + (Vt-Vr)*np.random.random(src.shape)

# Simulation
# ----------
tf = 1000*ms
X = np.zeros((src.shape[0],tf/ms))
src.setup()
n = 0
for t in range(int(tf/ms)):
    # Update membrane potential
    src.evaluate(1*ms)
    # Reset where needed
    src.V = np.where(src.V > Vt, Vr, src.V)
    # Count spikes
    n += np.sum(src.S)
    # Record activities
    for i in range(src.shape[0]): X[i][t] = src.S[i]

# Visualization
# -------------
plt.figure(figsize=(12,6))
for i in range(src.shape[0]):
    x = np.where(X[i] == 1)[0]
    y = np.ones((x.shape[0],))*i
    plt.plot(x, y, 'o', color='b')
plt.ylim(0, src.shape[0])
plt.xlim(0, tf/ms)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron number')
plt.title('Total number of spikes : %d' % int(n))
plt.show()
