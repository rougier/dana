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

source = ones(10, 'V')
target = zeros(10, 'V; I')
C = DenseConnection(source('V'), target('I'), np.ones(1),
                    'dW/dt = pre.V')
run(n=1)
