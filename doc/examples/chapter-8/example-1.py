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
run(n=3)
print G.V1, G.V2
print C1.weights, C2.weights
