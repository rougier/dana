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
import dana
import numpy as np
import matplotlib.pyplot as plt

ms  =  0.001   # millisecond
mV  =  0.001   # millivolt
taum = 20*ms
taue =  1*ms
taui = 10*ms
Vt    = 10*mV
Vr    =  0*mV

G = dana.zeros((1,), keys=['V', 'ge', 'gi'])
G.dV  = 'V + dt*(-V+ge-gi)/taum'
G.dge = 'ge + Ie - dt*ge/taue'
G.dgi = 'gi + Ii - dt*gi/taui'

G.constant = {'mV'  : mV,  'ms'  : ms,
              'taum' : taum,
              'taue' : taue,
              'taui' : taui,
              'Vt'  : Vt,
              'Vr'  : Vr}

Input = dana.zeros((1,), keys=['e','i'])
G.connect(Input.e, np.ones((1,))*3*mV, 'Ie')
G.connect(Input.i, np.ones((1,))*3*mV, 'Ii')

spiketimes = [('e',1*ms), ('e',10*ms), ('i',40*ms),
              ('e',50*ms), ('e',55*ms)]

tf = 100*ms
dt = 0.1*ms
V = []
for i in range(int(tf/dt)):
    Input.e = Input.i = 0
    for key,time in spiketimes:
        if time == i*dt:
            Input[key] = 1
    G.compute(dt)
    V.append(G.V[0])
plt.plot(V)
plt.show()


