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
Implementation of the Oja learning rule for extracting the principal component
of an elliptical gaussian distribution. Given that the distribution is
elliptical, its principal component should be oriented along the main axis of
the distribution, therefore, final weights should be +/-cos(theta), sin(theta)


References:
-----------

  E. Oja, "A Simplified Neuron Model as a Principal Component Analyzer"
  Journal of Mathematical Biology 15: 267-273, 1982.

'''
from numpy import *
from dana import *

def sample(theta, mu1, std1, mu2, std2):
    ''' Random sample according to an elliptical Gaussian distribution'''
    u1 = random.random()
    u2 = random.random()
    T1 = sqrt(-2.0*log(u1))*cos(2.0*pi*u2)
    T2 = sqrt(-2.0*log(u1))*sin(2.0*pi*u2)
    x = mu1 + (std1*T1*cos(theta) - std2*T2*sin(theta))
    y = mu2 + (std1*T1*sin(theta) + std2*T2*cos(theta))	
    return np.array([x,y])

theta = -135.0 * pi / 180.0
src = Group((2,), 'V = sample(theta,0.0,1.0,0.0,0.5)')
tgt = Group((1,), 'V')
C = DenseConnection(src('V'), tgt('V'), np.ones((1,2)),
                    'dW/dt = post.V*(pre.V-post.V*W)')
run(time=10.0,dt=0.001)
print "Learned weights : ", C.weights[0]
print "(should be +/- [%f, %f])" % (cos(theta), sin(theta))
