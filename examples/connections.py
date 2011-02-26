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
