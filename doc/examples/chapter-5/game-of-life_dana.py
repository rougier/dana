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

Z = Group((4,4), '''V = maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V)) : int;
                    N : int''')
Z.V = [[0,0,1,0], [1,0,1,0], [0,1,1,0], [0,0,0,0]]
SparseConnection(Z('V'), Z('N'), np.array([[1,1,1],[1,np.NaN,1],[1,1,1]]))
print 'Initial state:\n', Z.V
run(n=4)
print 'Final state:\n', Z.V
