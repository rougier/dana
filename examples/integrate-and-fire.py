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

@after(clock.tick)
def spike(time):
    global n

    # Reset where needed
    src.V = np.where(src.V > Vt, Vr, src.V)

    # Count spikes
    n += np.sum(src.S)

    # Record activities
    for i in range(src.shape[0]):
        X[i][int(time/clock.dt)] = src.S[i]

run(1*second,dt=1*millisecond)


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
