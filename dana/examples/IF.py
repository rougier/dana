#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009, 2010 Nicolas Rougier - INRIA - CORTEX Project
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
import matplotlib.pyplot as plt
import numpy as np
import dana

ms  =  0.001   # millisecond
mV  =  0.001   # millivolt
tau = 20.0*ms  # membrane time constant
Vt  =-50.0*mV  # spike threshold
Vr  =-60.0*mV  # reset value
El  =-49.0*mV  # resting potential
psp =  0.5*mV  # postsynaptic potential size
sparseness = 0.1

P = dana.zeros((50,), keys=['V','S'])
W = (np.random.random((P.shape+P.shape)) < sparseness).astype(int)
P.connect((P,'S'), W, 'I', shared=False)
P.dV = '-V+np.where(V > Vt, Vr, V-dt*(V-El)/tau + I*0.5*mV)'
P.dS = '-S +(V > Vt)'

# Initialize membrane potential
P.V = Vr+(Vt-Vr)*np.random.random(P.shape)

tf = 500*ms
n = 0
X = np.zeros((P.shape[0],tf/ms))
for t in range(int(tf/ms)):

    # Update membrane potential
    P.compute(1*ms)

    # Count spikes
    n += np.sum(P.S)

    # Record activities
    for i in range(P.shape[0]):
        X[i][t] = P.S[i]

for i in range(P.shape[0]):
    x = np.where(X[i] == 1)[0]
    y = np.ones((x.shape[0],))*i
    plt.plot(x, y, 'o', color='b')
plt.ylim(0, P.shape[0])
plt.xlim(0, tf/ms)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron number')
plt.title('Total number of spikes : %d' % int(n))
plt.show()
