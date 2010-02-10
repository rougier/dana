#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import dana, pylab

ms  =  0.001   # millisecond
mV  =  0.001   # millivolt
tau_a =  1*ms
tau_b = 10*ms
Vt    = 10*mV
Vr    =  0*mV
tf = 10*ms
dt = 0.1*ms

G = dana.zeros((1,), keys=['Va', 'Vb', 'S'])
Input = dana.zeros((1,), keys=['a','b'])

G.connect((Input,'a'), np.ones((1,))*6*mV, 'Ia')
G.connect((Input,'b'), np.ones((1,))*3*mV, 'Ib')

G.dVa = 'Ia - dt*Va/tau_a'
G.dVb = 'Ib - dt*Vb/tau_b'

spiketimes = [('a',1*ms),  ('b',2*ms), ('a',4*ms), ('b',3*ms)]
Va,Vb = [],[]
for i in range(int(tf/dt)):
    Input.a = Input.b = 0
    for key,time in spiketimes:
        if time == i*dt:
            Input[key] = 1
    G.compute(dt)
    Va.append(G.Va[0])
    Vb.append(G.Vb[0])
pylab.plot(Va)
pylab.plot(Vb)
pylab.show()


