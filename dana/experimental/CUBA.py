#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE
import time
import numpy as np
import dana
from random import sample
from scipy import random, sparse
import matplotlib.pyplot as plt


def sparse_random_matrix(dst_size, src_size, s0, s1, p=0.1, value=1.):
    W = sparse.lil_matrix((dst_size, src_size))
    for i in xrange(dst_size):
        k = random.binomial(s1-s0,p,1)[0]
        W.rows[i] = sample(xrange(s0,s1),k)
        W.rows[i].sort()
        W.data[i] = [value]*k
    return W

ms   =  0.001   # millisecond
mV   =  0.001   # millivolt
taum =  20*ms   # membrane time constant
taue =   5*ms   # excitatory synaptic time constant
taui =  10*ms   # inhibitory synaptic time constant
Vt   = -50*mV   # spike threshold
Vr   = -60*mV   # reset value
El   = -49*mV   # resting potential
we   = 1.62*mV  # excitatory synaptic weight
wi   = -9*mV    # inhibitory synaptic weight

G = dana.zeros((4000,), keys=['V','S','ge','gi'])
G.dV  = 'np.where(V>Vt, Vr, V+dt*(ge+gi-(V-El))/taum)'
G.dS  = 'V > Vt'
G.dge = 'Ie-dt*ge/taue'
G.dgi = 'Ii-dt*gi/taui'

# W = (sparse_random_matrix(len(G), len(G),    0, 3200, 0.02, we) + 
#      sparse_random_matrix(len(G), len(G), 3200, 4000, 0.02, wi))
# G.connect(G.S, W, 'I')
W = sparse_random_matrix(G.size, G.size,    0, 3200, 0.02, we)
G.connect(G.S, W, 'Ie')
W = sparse_random_matrix(G.size, G.size, 3200, 4000, 0.02, wi)
G.connect(G.S, W, 'Ii')

G.V = -60*mV+10*mV*np.random.rand(G.size)

t,dt = 500*ms, 0.1*ms
R = np.zeros((t/dt, len(G.S)))
t0 = time.clock()
V = []
for i in range(int(t/dt)):
    G.compute(dt)
    R[i] = G.S
    V.append(G.V[0])
print time.clock()-t0

for i in range(0,R.shape[0],10):
    x = np.where(R[i] == 1)[0]
    y = [i/float(R.shape[0])*t*10000,]*len(x)
    plt.plot(y, x, '.', color='b')
plt.xlabel('Time(ms)')
plt.ylabel('Neuron number')
plt.plot(V)
plt.show()

