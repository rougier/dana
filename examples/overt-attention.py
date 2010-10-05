#!/usr/bin/env python
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

'''
A Distributed Computational Model of Spatial Memory Anticipation During a
Visual Search Task. Anticipatory Behavior in Adaptive Learning Systems: From
Brains to Individual and Social Behavior, Springer - LNAI 4520, M.V. Butz and
O. Sigaud and G. Baldassarre and G. Pezzulo . (2006)

This script implements the model presented in [1]_ which is performing a
sequential search task with saccadic eye movements.

References
----------
    _[1] http://dx.doi.org/10.1007/978-3-540-74262-3_10
'''

from dana import *
from display import *

# SigmaPiConnection object
# ------------------------
class SigmaPiConnection(Connection):
    def __init__(self, source=None, modulator=None, target=None, scale=1.0, direction=1):
        Connection.__init__(self, source, target)
        self._scale = scale
        self._direction = +1
        if direction < 0:
            self._direction = -1
        names = modulator.dtype.names
        if names == None:
            self._actual_modulator = modulator
        else:
            self._actual_modulator = (modulator[names[0]])
    def output(self):
        src = self._actual_source
        mod = self._actual_modulator
        tgt = self._actual_target
        R = np.zeros(tgt.shape)
        if len(tgt.shape) == len(src.shape) == len(mod.shape) == 1:
            R = convolve1d(src,mod[::self._direction])
        elif len(tgt.shape) == len(src.shape) == len(mod.shape) == 2:
            R = convolve2d(src,mod[::self._direction,::self._direction])              
        else:
            raise NotImplemented
        return R*self._scale 


# Simulation parameters
# ---------------------
n  = 40
dt = 0.5
stimuli_size = 0.13
noise_level = 0.05

# Build groups
# ------------
visual = np.zeros((n,n))
tau_f = 1.0/3.0
focus = Group((n,n), '''dU/dt = tau_f*(-U + Ii + Iwm + L - 0.05) : float
                        V = minimum(maximum(U,0),1) : float
                        Ii: float; Iwm: float; L: float''')
tau_w = 1.0/1.5
wm = Group((n,n), '''dU/dt = tau_w*(-U + L + Ii + If + Ia + It -0.25) : float
                     V = minimum(maximum(U,0),1) : float
                     Ii: float; If: float; Ia: float; It: float; L: float''')
thal_wm = Group((n,n), '''dU/dt = (-U + L + I) : float
                          V = minimum(maximum(U,0),1) : float
                          I : float; L : float''')
anticipation = Group((n,n), '''dU/dt = (-U + I) : float
                               V = minimum(maximum(U,0),1) : float
                               I : float; L : float''')

# Connections
# -----------
s = (2*n+1,2*n+1)
K = 0.13*gaussian(s,2.83/n)-0.046*gaussian(s, 17.68/n); K[n,n] = 0.0
SharedConnection(visual,     focus('Ii'),  +0.018*gaussian(s, 1.42/n))
SharedConnection(focus('V'), focus('L'),   K)
SharedConnection(wm('V'),    focus('Iwm'), -0.005*gaussian(s, 4.24/n))
K = 0.185*gaussian(s, 1.77/n)-0.11*gaussian(s, 2.83/n); K[n,n] = 0.0
SharedConnection(visual,            wm('Ii'), +0.021*gaussian(s, 1.42/n) )
SharedConnection(focus('V'),        wm('If'), +0.023*gaussian(s, 1.42/n) )
SharedConnection(thal_wm('V'),      wm('It'), +0.195*gaussian(s, 1.06/n) )
SharedConnection(anticipation('V'), wm('Ia'), +0.023*gaussian(s, 1.42/n) )
SharedConnection(wm('V'),           wm('L'),  K)
SharedConnection(wm('V'), thal_wm('I'), 0.195*gaussian(s, 1.06/n) )
SigmaPiConnection(wm('V'), focus('V'), anticipation('I'),scale=0.05)
K = 0.4*gaussian(s, 2.12/n) - 0.2*gaussian(s, 2.83/n); K[n,n] = 0.0
SharedConnection(anticipation('V'), anticipation('L'), K)


def evaluate(epochs):
    global visual, stimuli, stimuli_size
    for i in range(epochs):
        encode(visual, stimuli, stimuli_size)
        run(t=dt,dt=dt)
        update()
        plt.draw()

def decode(Z):
    s = Z.sum()   
    if s == 0:
        return 0, 0
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    x = (Z.sum(axis=0)*np.linspace(xmin,xmax,Z.shape[1])).sum()
    y = (Z.sum(axis=1)*np.linspace(ymin,ymax,Z.shape[0]).T).sum()
    return y/s, x/s

def encode(Z, stimuli, size):
    Z[...] = 0
    for i in range(stimuli.shape[0]):
        Z[...] += gaussian(Z.shape, size, (stimuli[i][0],stimuli[i][1]))
    Z += np.random.uniform(-noise_level, noise_level, Z.shape)
    Z = np.maximum(np.minimum(Z,1),0)

def make_saccade(command, output, stimuli, size):
    stimuli -= decode(command)
    encode(output, stimuli, size)

def demo():
    global focus, visual, stimuli, stimuli_size
    for i in range(3):
        evaluate(100)
        make_saccade(focus['V'], visual, stimuli, stimuli_size)
    evaluate(100)


# Set visual
# ----------
stimuli = np.array([[0.5*np.sin(0.0*np.pi/3.0), 0.5*np.cos(0.0*np.pi/3.0)],
                    [0.5*np.sin(2.0*np.pi/3.0), 0.5*np.cos(2.0*np.pi/3.0)], 
                    [0.5*np.sin(4.0*np.pi/3.0), 0.5*np.cos(4.0*np.pi/3.0)]])

# Visualization
# -------------
plt.ion()
fig = plt.figure(figsize=(8,10),facecolor='white')
plot(plt.subplot(321), thal_wm('V'), 'Thal_Wm')
plot(plt.subplot(322), anticipation('V'), 'Anticipation')
plot(plt.subplot(323), wm('V'), 'Wm')
plot(plt.subplot(324), focus('V'), 'Focus')
plot(plt.subplot(325), visual, 'Visual')
plt.connect('button_press_event', button_press_event)
plt.draw()
demo()
plt.show()

