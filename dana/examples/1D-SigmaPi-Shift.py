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
import pylab
import matplotlib.pyplot as plt
from sigmaPiGroup import sigmaPiGroup


print "To test this script, run step_motor(200)"

# Simulation parameters
# ______________________________________________________________________________
n       = 150
dt      = 1.0
tau     = 1.0

# Build groups
# ______________________________________________________________________________
input = dana.zeros((n,), name='input')
motor = dana.zeros((n,), name='motor')
output = sigmaPiGroup(shape=(n,), name = 'output')

#output = dana.zeros((n,n), name='output')

# Connections
# ______________________________________________________________________________

# Shift the input activties in the opposite direction of the motor command
# Non optimized 
#output.connect_sigmapi(input.V, motor.V, -1.0, 1.0, n/2.0, 0.05,'S')
# Optimized 
#output.connect_sigmapi(input.V, motor.V, -1.0, 1.0, n/2.0, 0.05,'S#')

# Shift the input activities in the same direction as the motor command
# Non optimized
#output.connect_sigmapi(input.V, motor.V, 1.0, -1.0, n/2.0, 0.05,'S')
# Optimized
output.connect_sigmapi(input.V, motor.V, 1.0, -1.0, n/2.0, 0.08,'S*')


# Set Dynamic Neural Field equation
# ______________________________________________________________________________
output.dV = '-V + S'

# Set input
# ______________________________________________________________________________

input.V  = dana.gaussian(input.shape, 30.0/input.shape[0], (0.5,0.0))
input.V += dana.gaussian(input.shape, 30.0/input.shape[0], (-0.5,0.0))
# As the sigma pi is involving a convolution, the high frequency noise can
# be filtered out
input.noise = 0.2*numpy.random.random(input.shape)
input.V += input.noise

motorPosition = 0.0
motorAngularSpeed = 2.0 / n
# Make a motor command narrower than the inputs
# The convolution involved in the sigma pi spreads the input
motor.V = dana.gaussian(motor.shape, 10.0/motor.shape[0], motorPosition )

# Display result using pylab
# __________________________________________________________________________

main_fig = plt.figure(facecolor='white')
ax_input = main_fig.add_subplot(311)
plt.xlabel('Space',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.title('Input',fontsize=16)
X = numpy.arange(0, n, 1)
ax_input.plot(X,input.V, linewidth=1, color=(.1,.1,1), linestyle='-')
plt.xlim(0, input.shape[0])
plt.ylim(0, 1.2)

ax_motor = main_fig.add_subplot(312)
plt.xlabel('Space',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.title('Motor command',fontsize=16)
X = numpy.arange(0, n, 1)
ax_motor.plot(X,motor.V, linewidth=1, color=(.1,.1,1), linestyle='-')
plt.xlim(0, motor.shape[0])
plt.ylim(0, 1.2)


ax_output = main_fig.add_subplot(313)
plt.xlabel('Space',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.title('Output',fontsize=16)
X = numpy.arange(0, n, 1)
ax_output.plot(X,output.V, linewidth=1, color=(.1,.1,1), linestyle='-')
plt.xlim(0, output.shape[0])
plt.ylim(0, 1.2)

plt.draw()

# Define a step method to modify the motor signal
# ______________________________________________________________________________

def step_motor(n):
    global motorPosition, motorAngularSpeed
    for i in range(n):
        input.V -= input.noise
        input.noise = 0.2*numpy.random.random(input.shape)
        input.V += input.noise
        # Update the position of the motor signal
        motorPosition = motorPosition + dt * motorAngularSpeed
        # The bounds for motorPosition are specific to the way DANA deals the indexes
        if( (motorPosition >= 1.0) or (motorPosition <= -1.0)):
            motorAngularSpeed *= -1.0
        motor.V = dana.gaussian(motor.shape, 10.0/motor.shape[0], motorPosition)
        # Compute the output
        output.compute(dt)
        # Refresh the view
        update_display()

def update_display():
    X = numpy.arange(0, n, 1)
    plt.axes(ax_input)
    plt.cla()
    plt.plot(X,input.V, linewidth=1, color=(.1,.1,1), linestyle='-')
    plt.xlim(0, input.shape[0])
    plt.ylim(0, 1.2)

    plt.axes(ax_motor)
    plt.cla()
    plt.plot(X,motor.V, linewidth=1, color=(.1,.1,1), linestyle='-')
    plt.xlim(0, motor.shape[0])
    plt.ylim(0, 1.2)

    plt.axes(ax_output)
    plt.cla()
    plt.plot(X,output.V, linewidth=1, color=(.1,.1,1), linestyle='-')
    plt.xlim(0, output.shape[0])
    plt.ylim(0, 1.2)
    
    plt.draw()

plt.show()

