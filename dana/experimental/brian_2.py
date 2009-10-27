#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#   ____  _____ _____ _____ 
#  |    \|  _  |   | |  _  |   DANA, Distributed Asynchronous Adaptive Numerical
#  |  |  |     | | | |     |         Computing Framework
#  |____/|__|__|_|___|__|__|         Copyright (C) 2009 INRIA  -  CORTEX Project
#                         
#  This program is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free Software
#  Foundation, either  version 3 of the  License, or (at your  option) any later
#  version.
# 
#  This program is  distributed in the hope that it will  be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public  License for  more
#  details.
# 
#  You should have received a copy  of the GNU General Public License along with
#  this program. If not, see <http://www.gnu.org/licenses/>.
# 
#  If you modify  this software, you should include a notice  giving the name of
#  the person  performing the  modification, the date  of modification,  and the
#  reason for such modification.
# 
#  Contact: 
# 
#      CORTEX Project - INRIA
#      INRIA Lorraine, 
#      Campus Scientifique, BP 239
#      54506 VANDOEUVRE-LES-NANCY CEDEX 
#      FRANCE
# 
import numpy as np
import dana, pylab

ms  =  0.001   # millisecond
mV  =  0.001   # millivolt
tau_a =  1*ms
tau_b = 10*ms
Vt    = 10*mV
Vr    =  0*mV

G = dana.zeros((1,), keys=['Va', 'Vb', 'S'])
G.dVa = 'Va + Ia - dt*Va/tau_a'
G.dVb = 'Vb + Ib - dt*Vb/tau_b'
G.constant = {'mV'  : mV,  'ms'  : ms,
              'tau_a' : tau_a,
              'tau_b' : tau_b,
              'Vt'  : Vt,
              'Vr'  : Vr}

Input = dana.zeros((1,), keys=['a','b'])
G.connect(Input.a, np.ones((1,))*6*mV, 'Ia')
G.connect(Input.b, np.ones((1,))*3*mV, 'Ib')

spiketimes = [('a',1*ms),  ('b',2*ms), ('a',4*ms), ('b',3*ms)]

tf = 10*ms
dt = 0.1*ms
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


