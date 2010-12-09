#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
This example shows various connections and how to display them.
'''
from dana import *

n = 20
p = 2*n+1
A, B = Group((n,n),'V'), Group((n,n),'V')
C, D = Group((n,n),'V'), Group((n/2,n/2),'V')

for Z in [A,B,C,D]:
    Z.V = rnd.random(Z.shape)


# Difference of Gaussians connections
K = 1.25*gaussian((p,p),0.1) - 0.75*gaussian((p,p),1.0)
DenseConnection(B,A,K)
SharedConnection(C,A,K)
SparseConnection(D,A,K)

# Row, columns and point connections
DenseConnection(A,B, np.ones((1,1)))
DenseConnection(C,B, np.ones((1,p)))
DenseConnection(D,B, np.ones((p,1)))

# Random connections
SparseConnection(A, C, rnd.random((p,p)) * (rnd.random((p,p)) > .8))
SparseConnection(B, C, rnd.random((n,n)) * (rnd.random((n,n)) > .8))
SparseConnection(D, C, rnd.random((n/2,n/2)) * (rnd.random((n/2,n/2)) > .8))


if __name__ == '__main__':
    from display import *

    plt.figure(figsize=(10,10))
    plot(plt.subplot(2,2,1), A('V'), 'A')
    #'Difference of Gaussians using\n dense, shared and sparse connection')
    plot(plt.subplot(2,2,2), B('V'), 'B')
    #'Row, column and point connections')
    plot(plt.subplot(2,2,3), C('V'), 'C')
    #'Random sparse connections')
    plot(plt.subplot(2,2,4), D('V'), 'D')
    plt.connect('button_press_event', button_press_event)
    plt.show()
