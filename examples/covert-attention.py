#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
This is a demonstration of covert attention using dynamic neural fields. The
goal for the model is to focus successively on each ot the three stimuli. As
soon as a stimulus if focused, it entered the working memory where a virtual
"link" is made with the visual input. The focus may be considered as a gate
allowing the working memory to make a specific bind ith the visual input.

A switch is made by setting the reward group to a very high value that
disinhibit the striatum group which in turn inhibit the focus group. Since the
working memory is also continuously inhibiting the focus group, the focus can
only be made on never focused stimuli.

If you run the demo (python covert-attention), you will see three switched and
then you'll be allowed to click on any group to see different connections
between groups.
'''
from dana import *
from functools import partial

def format_coord(axis, x, y):
    Z = axis.get_array()
    if x is None or y is None or Z is None:
        return ''
    x,y = int(x), int(y)
    if 0 <= x < Z.shape[1] and 0 <= y < Z.shape[0]:
        return '[%d,%d]: %s' % (x,y, Z[y,x])
    return ''
def update(G=None, x=-1, y=-1):
    mgr = plt.get_current_fig_manager()
    if G is None:
        for axis,Z in mgr.subplots:
            axis.set_data (Z)
    else:
        for axis,z in mgr.subplots:
            axis.set_data(np.empty_like(z)*np.NaN)
        if hasattr(G, '_connections'):
            for C in G._connections:
                for axis,z in mgr.subplots:
                    if C._actual_source is z:
                        axis.set_data(C[y,x])
    plt.draw()
def button_press_event(event):
    G,x,y = None, -1, -1
    if event.inaxes and event.button == 1:
        G = event.inaxes.group
        x,y = int(event.xdata), int(event.ydata)
    update(G, x, y)
def plot(subplot, data, name):
     mgr = plt.get_current_fig_manager()
     a,b = 0.75, 1.0
     chessboard = np.array(([a,b]*16 + [b,a]*16)*16)
     chessboard.shape = 32,32
     if isinstance(data, Group):
         group = data.base
         data = data[data.keys[0]]
     else:
         group = data
         data = data
     plt.imshow(chessboard, cmap=plt.cm.gray, interpolation='nearest',
                extent=[0,group.shape[0],0,group.shape[1]], vmin=0, vmax=1)
     plt.hold(True)
     axis = plt.imshow(data, interpolation='nearest', cmap= plt.cm.PuOr_r,
                       origin='lower', vmin=-1, vmax=1,
                       extent=[0,group.shape[0],0,group.shape[1]])
     subplot.format_coord = partial(format_coord, axis)
     subplot.group = group
     if not hasattr(mgr, 'subplots'):
         mgr.subplots = []
     mgr.subplots.append((axis,data))
     subplot.text(.05, .05, name, fontsize=16,transform=subplot.transAxes)
     plt.hold(False)


# Simulation parameters
# ---------------------
n = 40
dt = 0.5
r = 0.7
theta = np.array([0.0 , 2.0 * np.pi/3.0 , - 2.0 * np.pi/3.0])
dtheta = np.pi/300.0

# Build groups
# ------------
visual = np.zeros((n,n))
focus = Group((n,n), '''dV/dt = -V+(L+Iv+Is)/30 -0.05 : float
                        U = maximum(V,0) : float
                        L : float; Iv: float; Is: float''')
wm  = Group((n,n), '''dV/dt = -V+(L+Iv+If)/31 - 0.2 : float
                      U = minimum(maximum(V,0),1) : float
                      L : float; Iv: float; If: float''')
striatum = Group((n,n), '''dV/dt = -V+(L+Iw+Ir)/28 - 0.3 : float
                           U = maximum(V,0) : float
                           L : float; Iw: float; Ir: float''')
reward = Group((1,1), '''dV/dt = -0.1*V : float
                         U = maximum(V,0) : float''')

# Connections
# -----------
s = (2*n+1,2*n+1)
SharedConnection(visual,        focus('Iv'),    +0.25*gaussian(s, 0.05))
SharedConnection(striatum('U'), focus('Is'),    -0.20*gaussian(s, 0.10))
SharedConnection(focus('U'),    focus('L'),     +1.70*gaussian(s, 0.10)
                                                -0.65*gaussian(s, 1.00))
SharedConnection(visual,        wm('Iv'),       +0.35*gaussian(s, 0.05))
SharedConnection(focus('U'),    wm('If'),       +0.20*gaussian(s, 0.05))
SharedConnection(wm('U'),       wm('L'),        +3.00*gaussian(s, 0.05)
                                                -0.50*gaussian(s, 0.10))
SharedConnection(wm('U'),       striatum('Iw'), +0.50*gaussian(s, 0.0625))
DenseConnection(reward('U'),    striatum('Ir'), +1.0)
SharedConnection(striatum('U'), striatum('L'),  +2.50*gaussian(s, 0.05)
                                                -1.00*gaussian(s, 0.10))


# Visualization
# --------------
plt.ion()
fig = plt.figure(figsize=(8,12), facecolor='white')
plot(plt.subplot(3,2,1), visual, 'Visual')
plot(plt.subplot(3,2,2), focus('U'), 'Focus')
plot(plt.subplot(3,2,3), wm('U'), 'Working memory')
plot(plt.subplot(3,2,4), striatum('U'), 'Striatum')
plot(plt.subplot(3,2,5), reward('U'), 'Reward')
plt.connect('button_press_event', button_press_event)


def reset():
    visual[...] = 0
    focus[...] = 0
    wm[...] = 0
    striatum[...] = 0
    reward[...] = 0
def switch():
    reward['V'] = 30.0
def rotate():
    global theta, visual
    theta += dtheta
    visual[...] = 0
    for i in range(theta.size):
        x, y = r*np.cos(theta[i]), r*np.sin(theta[i])
        visual += gaussian((n,n), 0.2, (x,y))
    visual += (2*rnd.random((n,n))-1)*.05
def iterate(t=100):
    for i in range(t):
        rotate()
        run(t=.5,dt=.5)
        update()
        plt.draw()
def demo(t=1000):
    reset()
    iterate(100)
    for i in range(2):
        switch()
        iterate(150)

demo()
plt.show()
