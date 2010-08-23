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

print "########################################"
print "To test this script, run step_motor(200)"
print "########################################"

# Simulation parameters
# ______________________________________________________________________________
n       = 30
dt      = 1.0
tau     = 1.0
h       = 0.0

# Build groups
# ______________________________________________________________________________
input = dana.zeros((n,n), name='input')
motor = dana.zeros((n,n), name='motor')
output = sigmaPiGroup(shape=(n,n), name = 'output')

#output = dana.zeros((n,n), name='output')

# Connections
# ______________________________________________________________________________

# Shift the input activties in the opposite direction of the motor command
# Non optimized 
#output.connect_sigmapi(input.V, motor.V, -1.0, 1.0, n/2.0, 0.05,'S')
# Optimized 
output.connect_sigmapi(input.V, motor.V, -1.0, 1.0, n/2.0, 0.05,'S#')

# Shift the input activities in the same direction as the motor command
# Non optimized
#output.connect_sigmapi(input.V, motor.V, 1.0, -1.0, n/2.0, 0.05,'S')
# Optimized
#output.connect_sigmapi(input.V, motor.V, 1.0, -1.0, n/2.0, 0.05,'S*')


# Set Dynamic Neural Field equation
# ______________________________________________________________________________
output.dV = '-V + maximum(V+dt/tau*(-V + S + h),0)'

# Set input
# ______________________________________________________________________________

input.V  = 0.5*dana.gaussian(input.shape, 0.1*40/input.shape[0], ( 0.5, 0.5))
input.V += dana.gaussian(input.shape, 0.1*40/input.shape[0], (-0.5,-0.5))

motorPosition = numpy.array([0.5, 0.0])
motorAngularSpeed = 2.0 * numpy.pi / 200.0

motor.V = dana.gaussian(motor.shape, 0.1*40/motor.shape[0], (motorPosition[0]*numpy.cos(motorPosition[1]) , motorPosition[0]*numpy.sin(motorPosition[1])))

# Define a step method to modify the motor signal
# ______________________________________________________________________________

def step_motor(n):
    global motorPosition, motorAngularSpeed
    for i in range(n):
        # Update the position of the motor signal
        motorPosition = motorPosition + [0.0, dt * motorAngularSpeed]
        motor.V = dana.gaussian(motor.shape, 0.1*40/motor.shape[0], (motorPosition[0]*numpy.cos(motorPosition[1]) , motorPosition[0]*numpy.sin(motorPosition[1])))
        # Compute the output
        output.compute(dt)
        # Refresh the view
        view.update()

# Display result using pylab
# __________________________________________________________________________
view = dana.pylab.view([ [input.V, output.V],[motor.V] ])
view.show()
