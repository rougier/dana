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
#
# Wilson, H.R. and Cowan, J.D., "Excitatory and inhibitory interactions in
# localized populations of model neurons, 1972, Biophysical Journal,
# 12:1-24. [1]
#
# The present script solve the 2 non-linear differential equations system.
#
#  * E(t) is the proportion of excitatory cells firing per unit time at the
#         instant t.
#
#  * I(t) is the proportion of inhibitory cells firing per unit time at the
#          instant t.
#
#  * c1, c3 - c2, c4 represent the average number of excitatory and inhibitory
#                     synapses per cell, respectively.
#
#  * P(t) is the external input to the excitatory subpopulation.
#
#  * Q(t) is the external input to the inhibitory subpopulation.
#
#  * tau is a time constant indicates refractory period.
#
# Author: Georgios Detorakis, CORTEX Team, INRIA Research Center, Nancy. 
# e-mail: gdetor@gmail.com / georgios.detorakis@inria.fr
# -----------------------------------------------------------------------------
from dana import *

# Interval of integration, [a,b].
a = 0.0
b = 300.0
# Time step of integration.
h = 0.001
# Number of nodes.
M = int( np.floor(( b - a+1)/h ))

# Definitions of necessary arrays. x is corresponding to E, y to I, p to P.
x = np.zeros((M,))
y = np.zeros((M,))
p = np.zeros((M,))

# Definitions of initial conditions.
E = 0.0
I = 0.0
# Initialization of time. 
t = 0
# Defintition of time constant.
tau = 10.0
#Defintitions of parameters.
re = 1.0
ri = 1.0
kn = 1.0
k = 1.0
# Definitions of connectivity coefficients.
c1 = 15.0
c2 = 15.0
c3 = 15.0
c4 = 3.0
# Initialization of P(t) and Q(t).
Q = 0
P = 0

# Model definition. Our model is a system of 2 non-linear equations.
model = Model(
   '''dE/dt = -E/tau + ((1-re*E)*(1/(1+np.exp(-(k*c1*E-k*c2*I+k*P-2))) - 1/(1 + np.exp(2*1.0))))/tau
      dI/dt = -I/tau + ((1-ri*I)*(1/(1+np.exp(-2*(kn*c3*E-kn*c4*I+kn*Q-2.5))) - 1/(1 + np.exp(2*2.5))))/tau
      P = 3*np.power(2,-0.03*t)''')


# Looping over time.
for i in range( M ):
    model.run( globals(), dt = h )
    x[i] = E
    y[i] = I
    p[i] = P
    t += h

# inverse sigmoid
def isigmoid( x, alpha, theta ):
    A = 1.0/( 1.0 + np.exp( alpha * theta ) )
    return ( -np.log( ( 1.0 - x - A )/( x + A ) ) )/alpha + theta

# Construction of Time axis.
T = np.linspace(a,b,M)

# Plot solutions E and I. 
plt.figure(figsize=(12,10))

plt.subplot(222)
plt.plot(T,x,'g',label='Ex')
plt.plot(T,y,'r',label='In')
plt.axvline(10,linestyle='--')
plt.title('Numerical Solution (DANA)')
plt.ylabel('E/I')
plt.legend(('Excitation','Inhibition'))

# Plot figure of phase plane.
plt.subplot(221)
plt.plot(y,x)
plt.title('Phase Plane and Isoclines')
plt.ylabel('E')

# Plot function P(t).
#plt.subplot(326)
##plt.plot(T,p,label='P')
#plt.xlabel('Time')
#plt.ylabel('P(t)')

# Plot solutions E, I and E-I according to [1](pg. 16).
plt.subplot(224)
plt.plot(T,x-y,'m',label='Ex-In')
plt.plot(T,x,'g',label='Ex')
plt.plot(T,y,'r',label='In')
plt.axvline(10,linestyle='--')
plt.ylabel('E/I')
plt.legend(('E-I','Excitation','Inhibition'))

# Isoclines. Evaluate the equations (13) and (14) from [1] at page 10.
plt.subplot(223)
U = (c1*x)/c2 - (isigmoid(x/(k-re*x), 1.0, 2.0)/c2 + p/c2)
V = (c4*y)/c3 + (isigmoid(y/(kn-ri*y), 2.0, 2.5)/c3 - Q/c3)
plt.plot(V,U)
plt.xlabel('I')
plt.ylabel('E')

# Shows all the plots. 
plt.show()
