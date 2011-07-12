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
'''
Reference: "Oscillatory Turing Patterns  in Reaction-Diffusion Systems with Two
           Coupled Layers", Lingfa Yang  and Irving R. Epstein, Physical Review
           Letters, volume 90, number 17, 2003.

Abstract: A model  reaction-diffusion system  with two  coupled  layers  yields
          oscillatory Turing patterns when  oscillation occurs in one layer and
          the  other supports  stationary Turing  structures.  Patterns include
          ‘‘twinkling eyes’’  where oscillating Turing spots are  arranged as a
          hexagonal lattice,  and localized  spiral or concentric  waves within
          spotlike  or   stripelike  Turing  structures.  A   new  approach  to
          generating the short-wave instability is proposed.

Website: http://hopf.chem.brandeis.edu/yanglingfa/pattern/oscTu/index.html

F₁(x,y) = 1/ε₁[x - x² - f₁z(x-q₁)/(x+q₁)]
F₂(x,y) = 1/ε₂[x - x² - f₂z(x-q₂)/(x+q₂)]
G(x,y) = x - z

∂x/∂t = Dx ∇²x + F₁(x,z) - 1/δ₁[x-r]    (1)
∂z/∂t = Dz ∇²z + G(x,z)                 (2)
∂r/∂t = Dr ∇²r + 1/δ₁[x-r] + 1/δ₂[u-r]  (3)
∂u/∂t = Du ∇²u + F₂(u,w) - 1/δ₂[u-r]    (4)
∂w/∂t = Dw ∇²w + G(u,w)                 (5)
'''
from dana import *

n = 100

Dx = Dz = Dr = 0.1
Dw = 100
q1 = q2 = 0.01

# Fig.3(a) 
Du,e1,f1,e2,f2 = 5, 0.14, 1.6, 0.4, 1.1
# Fig.3(b) 
# Du,e1,f1,e2,f2 = 10, 0.14, 1.6, 0.3, 0.7
# Fig.3(c) 
# Du,e1,f1,e2,f2 = 3, 0.215, 1.1, 0.5, 0.65

d1,d2 = 2.0*e1, 2.0*e2


def F1(x,z): return 1.0/e1 * (x - x**2 - f1*z*(x-q1)/(x+q1))
def F2(x,z): return 1.0/e2 * (x - x**2 - f2*z*(x-q2)/(x+q2))
def  G(x,z): return x - z
Z = Group((n,n), '''dx/dt = Dx*Lx + F1(x,z) - 1/d1*(x-r)    : float
                    dz/dt = Dz*Lz + G(x,z)                  : float
                    dr/dt = Dr*Lr + 1/d1*(x-r) + 1/d2*(u-r) : float
                    du/dt = Du*Lu + F2(u,w) - 1/d2*(u-r)    : float
                    dw/dt = Dw*Lw + G(u,w)                  : float
                    Lx; Lz; Lr; Lu; Lw; ''')
K = np.array([[np.NaN,  1., np.NaN], 
              [   1. , -4.,   1.  ],
              [np.NaN,  1., np.NaN]])
SparseConnection(Z('x'),Z('Lx'), K, toric=True)
SparseConnection(Z('z'),Z('Lz'), K, toric=True)
SparseConnection(Z('r'),Z('Lr'), K, toric=True)
SparseConnection(Z('u'),Z('Lu'), K, toric=True)
SparseConnection(Z('w'),Z('Lw'), K, toric=True)

Z['x'] = .1*np.random.random((n,n))
Z['z'] = .1*np.random.random((n,n))
Z['r'] = .1*np.random.random((n,n))
Z['u'] = .1*np.random.random((n,n))
Z['w'] = .1*np.random.random((n,n))


fig = plt.figure(figsize=(8,8))
border = 0.0
fig.subplots_adjust(bottom=border, top=1-border,
                    left=border, right=1-border)
im = plt.imshow(Z['x'], interpolation='bicubic', cmap=plt.cm.gray)
plt.xticks([]), plt.yticks([])

@clock.every(100*millisecond)
def frame(t):
    im.set_data(Z['x'])
    im.changed()
    plt.draw()
    # fig.savefig('/tmp/turing-movie-%08d.png' % (t*1000), dpi=25)

@clock.every(1*second)
def print_time(t):
    print 'Elapsed simulation time: %.2f seconds' % t

# @clock.every(10*second)
# def screenshot(t):
#     fig.savefig('/tmp/turing-screenshot-%08d.png' % (t*1000))

plt.ion()
run(time=200*second, dt=1*millisecond)
plt.ioff()
plt.show()
