#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
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
''' Numerical integration of dynamic neural fields.

This script implements the numerical integration of dynamic neural fields [1]_
of the form:

1 ∂U(x,t)             ⌠+∞
- ------- = -U(x,t) + ⎮  w(|x-y|).f(U(y,t)).dy + I(x,t) + h
α   ∂t                ⌡-∞

where U(x,t) is the potential of a neural population at position x and time t
      W(d) is a neighborhood function from ℝ⁺ → ℝ
      f(u) is the firing rate of a single neuron from ℝ → ℝ
      I(x,t) is the input at position x and time t
      h is the resting potential
      α is the temporal decay of the synapse

:References:
    _[1] http://www.scholarpedia.org/article/Neural_fields
'''
import scipy
from dana import *

#  Parameters
n       = 40
p       = 2*n+1
dt      = 0.05
alpha   = 15.0
tau     = 0.5
h       = 0.0
s       = (n*n)/(40.*40.)
noise   = 0.005
theta   = 0.000
dtheta  = 0.015 # rad/second
radius  = 0.750
second_stimulus = False

focus = zeros((n,n), '''dU/dt = (-U + (L+I+h)/alpha) / tau : float64;
                        V = np.maximum(U,0) : float64;
                        I : float64; L : float64''' )
SharedConnection(focus('V'), focus('L'),
                 (1.0*gaussian((p,p),0.1) - 0.5*gaussian((p,p),1.0))/s )

# Output decoding
X,Y = np.mgrid[0:n,0:n]
X,Y = 2*X/float(n-1) - 1, 2*Y/float(n-1) - 1

# Figure
fig = plt.figure(figsize=(14,7))
plt.ion()
plt.show()

axes_input = fig.add_subplot(1,2,1)
axes_focus = fig.add_subplot(1,2,2)
plt.figtext(0.5, 0.95,  'Dynamic Neural Field (%dx%d)' % (n,n),
            ha='center', color='black', weight='bold', size='large')



# This will be executed every 30*second
@clock.every(30*second)
def second_stimuli(*args):
    global second_stimulus
    second_stimulus = not second_stimulus


# This will be executed every 50 millisecond
@clock.every(50*millisecond)
def rotate_stimuli(*args):
    global theta, dtheta, radius
    global second_stimulus
    focus.I = 0.0

    # First stimulus
    x, y = radius*np.cos( theta ), radius*np.sin( theta )
    focus.I += gaussian((n,n),0.1,(x,y))

    # Second stimulus
    if second_stimulus:
        x, y = radius*np.cos(theta+np.pi), radius*np.sin(theta+np.pi)
        focus.I += 2.0*gaussian((n,n),0.2,(x,y))

    # Noise
    focus.I += (2*rnd.random((n,n))-1)*noise

    # Be careful with the update period
    theta += dtheta*(50*millisecond/(1*second))


# This will be executed every second
@clock.every(1*second)
def do_display(*args):
    global axes_input, axes_focus, X, Y

    I,V = focus.I, focus.V
    axes_input.cla()
    axes_input.set_title('Input', fontsize=20)
    axes_input.grid(True)
    axes_input.imshow(I, extent=(-1,+1,-1,+1), cmap=plt.cm.gray_r,
                      interpolation = 'nearest', origin='lower')
    axes_focus.cla()
    axes_focus.set_title('Focus', fontsize=20)
    axes_focus.grid(True)
    axes_focus.imshow(V,extent=(-1,+1,-1,+1), cmap=plt.cm.gray_r,
                      interpolation = 'nearest', origin='lower')
    x,y,s = 0,0,V.sum()
    if s:
        x,y = (X*V).sum()/s, (Y*V).sum()/s
        axes_input.plot([y],[x], marker='o', markerfacecolor='yellow', markersize=10)
    plt.draw()


# Run the model
run(time=1000, dt=dt)
