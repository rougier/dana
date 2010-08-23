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
'''
Numerical integration of dynamic neural fields
----------------------------------------------
This script implements the numerical integration of dynamic neural fields [1]_
of the form:
                  
1 ∂U(x,t)             1 ⌠+∞                        1 
- ------- = -U(x,t) + - ⎮   W(|x-y|)f(U(y,t)) dy + - I(x,t)
α   ∂t                τ ⌡-∞                        τ

where # U(x,t) is the potential of a neural population at position x and time t.
      # W(x) is a neighborhood function
      # f(x) is the firing rate of a single neuron.
      # α is the temporal decay of the synapse.
      # τ is a scaling term
      # I(x,t) is the input at position x.

References
----------
    _[1] http://www.scholarpedia.org/article/Neural_fields
'''

import numpy, dana
import matplotlib.pyplot as plt
from sigmaPiGroup import sigmaPiGroup
import gobject

# Simulation parameters
# ______________________________________________________________________________

n       = 100
dt      = 1.0 # ms

# Build groups
# ______________________________________________________________________________

input        = dana.zeros(shape=(n,), keys = ['V'], name='input')
focus        = dana.zeros(shape=(n,), keys = ['U', 'V'], name='focus')
wm           = dana.zeros(shape=(n,), keys = ['U', 'V'], name='wm')
anticipation = sigmaPiGroup(shape=(n,), keys = ['U', 'V'], name = 'anticipation')

# Connections
# ______________________________________________________________________________

# Connections of the focus map
# ___________________________

# Focus
Wi = 0.5*dana.gaussian((2*n+1,), 0.05)
focus.connect( input.V, Wi, 'I', shared=True)
#Wl = 1.7 * dana.gaussian((2*n+1,), 23.0/n) - 0.8 * dana.gaussian((2*n+1,), 230.0/n)
Wl = 15.0*(.85*dana.gaussian((2*n+1,), 0.05) - 0.75 * dana.gaussian((2*n+1,), 1.5))
focus.connect( focus.U, Wl, 'L', shared=True)
# Small inhibitory bias to prevent reselection of the previously selected targets 
# This bias is particularly usefull for the selection mechanism after a saccade
Wi = -0.035*dana.gaussian((2*n+1,), 11.53/n)
focus.connect (wm.U, Wi, 'Iw', shared=True)

# Set Dynamic Neural Field equation
# ______________________________________________________________________________

h_focus = 0.0
alpha_focus = 1.0;
mod_input = 1.0
tau_focus = 30.0
focus.dV = 'dt/tau_focus*(-V+(L*100.0/n+I*mod_input + Iw)/alpha_focus + h_focus)'
focus.dU = '-U + 1.0 / (1.0 + exp(-1.5*V))'

wm.V = .75*dana.gaussian((n,), 0.05) - 0.75 * dana.gaussian((n,), 1.0)

# Set input
# ______________________________________________________________________________

stimuli = numpy.linspace(-0.8, 0.8, 4).tolist()
varianceStimuli = 6.0/n

for s in stimuli:
    input.V  += dana.gaussian(n, varianceStimuli, s)

input.noise = 0.05*numpy.random.random(n)
input.V += input.noise


main_fig = plt.figure(facecolor='white')

ax_input = main_fig.add_subplot(221)
plt.xlabel('Space',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.title('Input',fontsize=16)
X = numpy.arange(0, n, 1)
ax_input.plot(X,input.V, linewidth=1, color=(.1,.1,1), linestyle='-')
plt.ylim(0,2)

ax_focus = main_fig.add_subplot(222)
plt.xlabel('Space',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.title('Focus',fontsize=16)
X = numpy.arange(0, n, 1)
ax_focus.plot(X,focus.V, linewidth=1, color=(.1,.1,1), linestyle='-')
plt.ylim (-2.0,2.0)

ax_wm = main_fig.add_subplot(223)
plt.xlabel('Space',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.title('Working memory',fontsize=16)
X = numpy.arange(0, n, 1)
ax_wm.plot(X,wm.V, linewidth=1, color=(.1,.1,1), linestyle='-')
plt.ylim(0,1.0)

ax_anticipation = main_fig.add_subplot(224)
plt.xlabel('Space',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.title('Anticipation',fontsize=16)
X = numpy.arange(0, n, 1)
ax_anticipation.plot(X,anticipation.V, linewidth=1, color=(.1,.1,1), linestyle='-')
plt.ylim(0,1.0)

plt.draw()

def step_compute(n):
    for i in range(n):
        input.V -= input.noise
        input.noise = 0.1*numpy.random.random(input.shape)
        input.V += input.noise

        input.compute(dt)
        focus.compute(dt)

        update_display()

def update_display():
    X = numpy.arange(0, n, 1)
    
    plt.axes(ax_input)
    plt.cla()
    ax_input.plot(X,input.V, linewidth=1, color=(.1,.1,1), linestyle='-')
    plt.ylim(0,1.2)
    
    plt.axes(ax_focus)
    plt.cla()
    ax_focus.plot(X,focus.U, linewidth=1, color=(.1,.1,1), linestyle='-')
    plt.ylim(0.0,1.0)

    plt.axes(ax_wm)
    plt.cla()
    ax_wm.plot(X,wm.V, linewidth=1, color=(.1,.1,1), linestyle='-')
    plt.ylim(-1.0,1.0)

    plt.axes(ax_anticipation)
    plt.cla()
    ax_anticipation.plot(X,anticipation.V, linewidth=1, color=(.1,.1,1), linestyle='-')
    plt.ylim(-1.0,1.0)

    plt.draw()

plt.show()


