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

#
# This script implements the model of
#
#    Izhikevich E.M. (2003) 
#    Simple Model of Spiking Neurons.
#    IEEE Transactions on Neural Networks, 14:1569- 1572
#
# The parameters are freely available in the MATLAB simulation of the author
# http://www.izhikevich.org/publications/izhikevich.m

import numpy as np
import dana, pylab

ms  =  0.001   # millisecond
mV  =  0.001   # millivolt
tau = 1.0*ms  # membrane time constant
Vt = 30*mV

# Parameters of the models

# 0  - Regular Spiking
# 1  - Intrinsically bursting
# 2  - Chattering
# 3  - Fast Spiking
neuronModel = 2

pars=[[0.02  ,  0.2   ,  -65  ,   8     ,  10  , 'Regular spiking (RS)'],
      [0.02  ,  0.2   ,  -55  ,   4     ,  10  , 'Intrinsically bursting (IB)'],
      [0.02  ,  0.2   ,  -50  ,   2     ,  10  , 'Chattering (CH)'],
      [0.1   ,  0.2   ,  -65  ,   2     ,  10  , 'Fast Spiking']]

a = pars[neuronModel][0]
b = pars[neuronModel][1]
c = pars[neuronModel][2]*mV
d = pars[neuronModel][3]*mV
I = pars[neuronModel][4]*mV

# Equations of the model of Izhikevich(2003)
#  v' = 0.04 v**2 + 5 v + 140 - u + I
#  u' = a (b v - u )
#  if v >= 30 mV
#     v <- c
#     u <- u + d

P = dana.zeros((1,), keys=['I','V','U','S'])
P.constant = {'mV'  : mV,  'ms'  : ms,
              'tau' : tau, 'Vt'  : Vt}

# Implements the condition : if v >= Vt, then V = c, else the dynamical equation
# To implement the first condition, we use the fact that the equation is updated with the Euler scheme : V(t+1) = V(t) + dV
# Therefore, to get V(t+1) = c, we define dV = -V(t) + c
P.dV = 'np.where(V >= Vt, -V + c, dt/tau*( 0.04*V**2/mV + 5*V + 140*mV - U + I))'
# Implements the condition : if v >= Vt, then U = U + d, else the dynamical equation
# As before, to implement the first condition, we write : U(t+1) = U(t) + dU, with dU = d
P.dU = 'np.where(V >= Vt, d, dt/tau*(a * (b * V - U)))' 

# Condition for emiting a spike : S(t+1) = S(t) + dS, with dS = (V >= Vt)?
P.dS = '-S +(V >= Vt)' # Spikes

P.dI = ''

# Initialize membrane potential
P['V'] = c
P['U'] = 0.0
P['I'] = 0.0

# Simulate for tf seconds
tf = 250*ms
# Inject the current at ti seconds
ti = 50*ms
# Simulation time step
dt =0.1*ms

times = np.linspace(0,tf,int(tf/dt)+1)
# History of the spikes
X = np.zeros((len(times),))
# Number of spikes
n = 0
# History of the 2 variables of the 2D model
Vhist = np.zeros((len(times),))
Uhist = np.zeros((len(times),))

# Let's evolve the system
for t in range(len(times)):
    # Update membrane potential
    if(times[t] <= ti):
        P['I'] = 0
    else:
        P['I'] = I
    P.compute(dt)
    # Count spikes
    n += np.sum(P.S)
    # Record activities
    X[t] = P.S[0]
    Vhist[t] = P.V[0]
    Uhist[t] = P.U[0]

# Plot the membrane potential function of time
pylab.figure()
pylab.subplot(1,2,1)
pylab.plot(times,Vhist)
pylab.xlabel('Time (s)')
pylab.ylabel('V (mv)')
pylab.title(pars[neuronModel][5])

# Indicate the spiking times
pylab.subplot(1,2,2)
pylab.plot(np.where(X == 1), np.ones(len(np.where(X==1))),'o',color='b')
pylab.xlabel('Time (s)')
pylab.ylabel('Spike')
pylab.title('Total number of spikes : %d' % int(n))

pylab.show()

