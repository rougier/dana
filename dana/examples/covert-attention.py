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


# Simulation parameters
# ______________________________________________________________________________
n       = 40
dt      = 0.5

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
focus.connect( visual['V'], 'I', Wi, shared=False)
Wl = 1.7 * gaussian(2*n+1, 0.1*n) - 0.65 * gaussian(2*n+1, 1.0 * n)
focus.connect( focus['V'], 'L', Wl, shared=False)
Wstr = -0.2 * gaussian(2*n+1, 0.1 * n)
focus.connect( striatum_inhib['V'], 'Istr', Wstr, shared=False)

# Wm
Wi = 0.25 * gaussian(2*n+1, 0.05*n)
wm.connect( visual['V'], 'I', Wi, shared=False)
Wf = 0.2* gaussian(2*n+1, 0.05*n)
wm.connect( focus['V'], 'If', Wf, shared=False)
Wl = 3.0 * gaussian(2*n+1, 0.05*n) - 0.5 * gaussian(2*n+1, 0.1*n)
wm.connect( wm['V'], 'L', Wl, shared=False)

# Striatum inhib
Wi = 0.5 * gaussian(2*n+1, 0.0625*n)
striatum_inhib.connect(wm['V'], 'I', Wi, shared=False)
Wir = 8.0 * numpy.ones((2*n+1, 2*n+1))
striatum_inhib.connect( reward['V'], 'Ir', Wi, shared=False)
Wl = 2.5 * gaussian(2*n+1, 0.05*n) - 1.0*gaussian(2*n+1, 0.1*n)
striatum_inhib.connect(striatum_inhib['V'], 'L', Wl, shared=False)


# Set Dynamic Neural Field equation
# ______________________________________________________________________________
focus.constant['tau'] = 0.75
focus.constant['alpha'] = 30.0
focus.constant['h'] = -0.05
focus.equation['V'] = 'maximum(V+dt/tau*(-V+(L+I+Istr)/alpha + h),0)'

wm.constant['tau'] = 0.75
wm.constant['alpha'] = 31.0
wm.constant['h'] = -.2
wm.equation['V'] = 'minimum(maximum(V+dt/tau*(-V+(L+If+I)/alpha + h),0),1.0)'

striatum_inhib.constant['tau'] = 0.75
striatum_inhib.constant['alpha'] = 28.0
striatum_inhib.constant['h'] = -0.3
striatum_inhib.equation['V'] = 'maximum(V+dt/tau*(-V+(L+I+Ir)/alpha+h),0)'

reward.constant['tau'] = 30.0
reward.equation['V'] = 'maximum(V+dt/tau*(-V),0)'

# Set input
# ______________________________________________________________________________
visual['V']  = gaussian(n, 0.1*n, ( -0.25, 0.25))
visual['V'] += gaussian(n, 0.1*n, (-0.25,-0.25))
visual['V'] += gaussian(n, 0.1*n, (0.25,0.0))
visual['V'] += (2*numpy.random.random((n,n))-1)*.05


# Display result using pylab
# __________________________________________________________________________
view = dana.pylab.view([ [visual['V'],focus['V']],
                         [wm['V'], striatum_inhib['V'], reward['V']] ])
view.show()

def switch():
    reward['V'] = 20.0

def run(t):
    for i in range(t):
        focus.compute(dt)
	wm.compute(dt)
	striatum_inhib.compute(dt)
	reward.compute(dt)
        view.update()
