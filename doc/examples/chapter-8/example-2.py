#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009,2010,2011 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
from dana import *

G = ones(1, '''V1 = I1; I1
               V2 = I2; I2''')
C1 = DenseConnection(G('V1'), G('I1'), np.ones(1) )
C2 = DenseConnection(G('V2'), G('I2'), np.ones(1), 'dW/dt = 1')

for i in range(4):
    print 't=%d: V₁(%d) = %f' % (i,i,G.V1[0])
    print '     V₂(%d) = %f' % (i,G.V2[0])
    print '     W₁(%d) = %f' % (i,C1.weights[0])
    print '     W₂(%d) = %f' % (i,C2.weights[0])
    print
    run(n=1)
