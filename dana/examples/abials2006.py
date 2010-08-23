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
#  Contact: 
# 
#      CORTEX Project - INRIA
#      INRIA Lorraine, 
#      Campus Scientifique, BP 239
#      54506 VANDOEUVRE-LES-NANCY CEDEX 
#      FRANCE
#
import time, numpy
import dana, dana.pylab
from sigmaPiGroup import sigmaPiGroup

# Simulation parameters
# ______________________________________________________________________________
n       = 40
dt      = 0.1
tau     = 1.0
h       = 0.0

# Build groups
# ______________________________________________________________________________
input        = dana.zeros((n,n), name='input')
focus        = dana.zeros((n,n), name='focus')
wm           = dana.zeros((n,n), name='wm')
anticipation = sigmaPiGroup(shape=(n,n), name = 'anticipation')

# Connections
# ______________________________________________________________________________

# Connections of the focus map
# ___________________________

# Excitation of the input
Wi = 0.01*dana.gaussian((2*n+1,2*n+1), 2.0/n)
focus.connect(input, Wi, 'I', shared=True)
# Global lateral connections for the competition
Wi = 0.122*dana.gaussian((2*n+1,2*n+1), 0.1) - 0.07*dana.gaussian((2*n+1,2*n+1), 1.0)
focus.connect (focus, Wi, 'L', shared=True)
# Small inhibitory bias to prevent reselection of the previously selected targets 
# This bias is particularly usefull for the selection mechanism after a saccade
Wi = -0.0001*dana.gaussian((2*n+1,2*n+1), 2.0/n)
focus.connect (wm, Wi, 'Iw', shared=True)

# Connections of the wm map
# _________________________
# Excitation from the input
Wi = 0.015*dana.gaussian((2*n+1,2*n+1), 2.0/n)
wm.connect(input, Wi, 'Ii', shared=True)
# Excitation from the focus map gating the entrance in working memory
Wi = 0.015*dana.gaussian((2*n+1,2*n+1), 2.0/n)
wm.connect(focus, Wi, 'If', shared=True)
# Excitation from the anticipation map allowing the post-saccadic update
Wi = 0.015*dana.gaussian((2*n+1,2*n+1), 2.0/n) # 0.013
wm.connect(anticipation, Wi, 'Ia', shared=True)
# 0.22 , 2.5 / n , 0.11, 3.5/n
# Local lateral connections providing the short term memory
Wi = (0.2*dana.gaussian((2*n+1,2*n+1), 2.4/n) - 0.12 * dana.gaussian((2*n+1,2*n+1), 3.0/n)) # 0.197 2.5/n ,  0.12  3.0/n
wm.connect(wm, Wi, 'L', shared=True)



# Connections of the anticipation map
# ___________________________________
# Shift the input activties in the opposite direction of the motor command
anticipation.connect_sigmapi(wm.V, focus.V, -1.0, 1.0, n/2.0, 0.07,'S#')
# Lateral connections to restrict the extent of the anticipatory signal
Wi = 0.0*(0.3*dana.gaussian((2*n+1,2*n+1), 3.0/n) - 0.2 * dana.gaussian((2*n+1,2*n+1), 4.0/n))
anticipation.connect(anticipation, Wi, 'L', shared=True)

# Set Dynamic Neural Field equation
# ______________________________________________________________________________
mod_input = 1.0
tau_focus = 1.0
focus.dV = '-V + maximum(V + dt/tau_focus * (-V + L + mod_input*I + Iw ),0)'

# Increase the dt of the wm and anticipation maps compared to the focus
tau_wm = 1.0
h_wm = -0.2
#wm.dV = '-V + minimum(maximum(V + dt/tau_wm * (-V + (L + Ii + If + Ia + h_wm)/1.1), 0),1.0)'
wm.dV = '-V + minimum(maximum(V + dt/tau_wm * (-V + (L + Ii + If + Ia + h_wm)/1.1), 0),1.0)'

tau_anticipation = 1.0
h_anticipation = -0.1
anticipation.dV = '-V + minimum(maximum(V + dt/tau_anticipation * (-V + L + S + h_anticipation),0),1.0)'


# Set input
# ______________________________________________________________________________

stimuli = [ [0.5, 0.0] , 
            [0.5*numpy.cos(2.0*numpy.pi/3.0), 0.5*numpy.sin(2.0*numpy.pi/3.0)], 
            [0.5*numpy.cos(4.0*numpy.pi/3.0), 0.5*numpy.sin(4.0*numpy.pi/3.0)]]
varianceStimuli = 0.13

for s in stimuli:
    input.V  += dana.gaussian((n,n), varianceStimuli, ( s[0], s[1]))

# Define a step method to modify the motor signal
# ______________________________________________________________________________

inputNoiseAmplitude = 0.02
inputNoise = numpy.random.random(input.shape) * inputNoiseAmplitude
input.V += inputNoise

def step_compute(n,clear=False):
    global inputNoise
    for i in range(n):
        print i
        input.V -= inputNoise
        inputNoise = numpy.random.random(input.shape) * inputNoiseAmplitude
        input.V += inputNoise
        wm.compute(dt)
        if(clear):
            focus.V = numpy.zeros(focus.shape)
        else:
            focus.compute(dt)
        anticipation.compute(dt)
        view.update()

def decode_focus_map():
    c = numpy.array([0.0, 0.0])
    cy = 0.0
    sum_act = 0.0
    for i in range(focus.shape[0]):
        for j in range(focus.shape[1]):
            c += focus.V[i,j] * numpy.array([i-focus.shape[0]/2.0,j - focus.shape[1]/2.0])
            sum_act += focus.V[i,j]
    if(sum_act != 0.0):
        c /= sum_act
    else:
        # Return invalid indexes
        c = numpy.array([-1, -1])
    return c

def shift_input(c):
    for i in range(len(stimuli)):
        stimuli[i] = stimuli[i] + c

def make_saccade():
    global inputNoise
    # Shift the input stimuli in the opposite direction of the saccade
    shift_input(-1.0 * numpy.multiply(decode_focus_map(), numpy.array([2.0 / focus.shape[0], 2.0/focus.shape[1]])))
    # Update the activities of the input map
    input.V = numpy.zeros(input.shape)
    for s in stimuli:
        input.V  += dana.gaussian((n,n), varianceStimuli, ( s[0], s[1]))
    # Add a random noise
    inputNoise = numpy.random.random(input.shape) * inputNoiseAmplitude
    input.V += inputNoise
    # Update the display
    view.update()

# Display result using pylab
# __________________________________________________________________________
#view = dana.pylab.view([ [thal_wm.V, anticipation.V],[wm.V, focus.V],[input.V] ])
view = dana.pylab.view([ [wm.V, anticipation.V],[input.V , focus.V]])
view.show()
