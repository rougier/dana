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

source = np.ones((3,3))
target = zeros((3,3),'V = V+I; I')
kernel = np.eye(9)
C = DenseConnection(source, target('I'), kernel)
run(n=5)
print target.V

