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

source = np.ones((2,2))
target = zeros((3,3),'I')
kernel = np.ones((target.size,source.size))
C = DenseConnection(source, target('I'), kernel)
C.propagate()
print target.I
