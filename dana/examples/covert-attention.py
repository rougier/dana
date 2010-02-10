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
import time, numpy
import dana, dana.pylab

# Simulation parameters
# ______________________________________________________________________________
n       = 40
dt      = 0.5
tau     = 1.0

# Gaussian function
# ______________________________________________________________________________
def gaussian(shape, width=(0.1,0.1), center=(0.0,0.0)):
    """ Computes a gaussian function of the form
        g(x,y) = exp(-((x-cx)²/wx²-(y-cy)²/wy²)/2) """
    if type(shape) in [int, float]:
        shape = (int(shape),int(shape))
    if type(width) in [int,float]:
        w = (float(width), float(width))
    else:
        w = (float(width[0]),float(width[1]))
    if type(center) in [int,float]:
        c = (shape[0]/2 + center*shape[0],
             shape[1]/2 + center*shape[1])
    else:
        c = (shape[0]/2 + center[0]*shape[0],
             shape[1]/2 + center[1]*shape[1])
    X,Y = numpy.mgrid[0:shape[0],0:shape[1]]
    return numpy.exp(-(((X-c[0])/w[0])**2 + ((Y-c[1])/w[1])**2)/2)

# Build groups
# ______________________________________________________________________________

visual = dana.group((n,n), name='visual')
focus = dana.group((n,n), name='focus')
wm = dana.group((n,n), name='wm')
striatum_inhib = dana.group((n,n), name='striatum_inhib')
reward = dana.group((1,1), name='reward')

# Connections
# ______________________________________________________________________________

# Focus
Wi = 0.25 * gaussian(2*n+1, 0.05 * n)
focus.connect( visual, Wi, 'I', shared=True)

Wl = 1.7 * gaussian(2*n+1, 0.1*n) - 0.65 * gaussian(2*n+1, 1.0 * n)
focus.connect( focus, Wl, 'L', shared=True)

Wstr = -0.2 * gaussian(2*n+1, 0.1 * n)
focus.connect( striatum_inhib, Wstr, 'Istr', shared=True)

# Wm
Wi = 0.3 * gaussian(2*n+1, 0.05*n)   # 0.25
wm.connect( visual, Wi, 'I', shared=True)

Wf = 0.2* gaussian(2*n+1, 0.05*n)
wm.connect( focus, Wf, 'If', shared=True)

Wl = 3.0 * gaussian(2*n+1, 0.05*n) - 0.5 * gaussian(2*n+1, 0.1*n)
wm.connect( wm, Wl, 'L', shared=True)

# Striatum inhib
Wi = 0.5 * gaussian(2*n+1, 0.0625*n)
striatum_inhib.connect(wm, Wi, 'I', shared=True)

Wir = 10.0 * numpy.ones((2*n+1, 2*n+1))
striatum_inhib.connect( reward, Wi, 'Ir', shared=False)

Wl = 2.5 * gaussian(2*n+1, 0.05*n) - 1.0*gaussian(2*n+1, 0.1*n)
striatum_inhib.connect(striatum_inhib, Wl, 'L', shared=True)

# Set Dynamic Neural Field equation
# ______________________________________________________________________________
h_focus = -0.05
alpha_focus = 30.0;
focus.dV = '-V+maximum(V+dt/tau*(-V+(L+I+Istr)/alpha_focus + h_focus),0)'

alpha_wm = 31.0
h_wm = -0.2
wm.dV = '-V+minimum(maximum(V+dt/tau*(-V+(L+If+I)/alpha_wm + h_wm),0),1.0)'

alpha_str = 28.0
h_str = -0.3
striatum_inhib.dV = '-V+maximum(V+dt/tau*(-V+(L+I+Ir)/alpha_str+h_str),0)'

rew_tau = 10.0
reward.dV = '-V+maximum(V+dt/rew_tau*(-V),0)'

# Set input
# ______________________________________________________________________________
r_stimuli = 0.30
theta_stimuli = [0.0 , 2.0 * numpy.pi / 3.0 , - 2.0 * numpy.pi / 3.0]

visual.V = numpy.zeros(visual.shape)
for theta in theta_stimuli:
    visual.V += gaussian(n, 0.1*n, (0.35 * numpy.cos(theta), r_stimuli * numpy.sin(theta)))
visual.V += (2*numpy.random.random((n,n))-1)*.05

# Display result using pylab
# __________________________________________________________________________
view = dana.pylab.view([ [wm.V, striatum_inhib.V, reward.V],
			 [visual.V,focus.V] ])
view.show()

# Some tool functions  (including the functions for the demos)
# __________________________________________________________________________

def reset():
  visual.V = numpy.zeros(visual.shape)
  focus.V = numpy.zeros(focus.shape)
  wm.V = numpy.zeros(wm.shape)
  striatum_inhib.V = numpy.zeros(striatum_inhib.shape)
  reward.V = numpy.zeros(reward.shape)
  view.update()

def switch():
  reward.V = 30.0

# To see the effect of rotate on the input, just try  rotate(numpy.pi/100.0, 100)
def rotate(angular_speed, t, update_view=True):
  global theta_stimuli, r_stimuli
  for i in range(t):
    visual.V = numpy.zeros(visual.shape)
    for j in range(len(theta_stimuli)):
      theta_stimuli[j] = theta_stimuli[j] + angular_speed
      visual.V += gaussian(n, 0.1*n, (0.35 * numpy.cos(theta_stimuli[j]), r_stimuli * numpy.sin(theta_stimuli[j])))
      visual.V += (2*numpy.random.random((n,n))-1)*.05
      if(update_view):
	view.update()

def run(t):
  for i in range(t):
    visual.V = numpy.zeros(visual.shape)
    for theta in theta_stimuli:
      visual.V += gaussian(n, 0.1*n, (0.35 * numpy.cos(theta), r_stimuli * numpy.sin(theta)))
      visual.V += (2*numpy.random.random((n,n))-1)*.05
    focus.compute(dt)
    wm.compute(dt)
    striatum_inhib.compute(dt)
    reward.compute(dt)
    view.update()

def run_rotation(angular_speed, t):
  for i in range(t):
    rotate(angular_speed,1,False)
    focus.compute(dt)
    wm.compute(dt)
    striatum_inhib.compute(dt)
    reward.compute(dt)
    view.update()  

def demo():
  reset()
  run(100)
  for i in range(2):
      switch()
      run(150)
	
def demo_rotation():
  reset()
  run(100)
  for i in range(3):
    switch()
    run_rotation(numpy.pi/300.0,100)
