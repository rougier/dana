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
Show influence of mask on group
'''
from dana import *

src = Group((3,3), 'V')
src.V = 1
src.mask = np.array([[1,1,1],
                     [1,0,1],
                     [1,1,1]])
src.V = 1
dst = Group((3,3), 'V:float')
setup()

print src
print

C = DenseConnection(src, dst, np.ones((3,3)))
print C.output()
print

C = SparseConnection(src, dst, np.ones((3,3)))
print C.output()
print

C = SharedConnection(src, dst, np.ones((3,3)))
print C.output()
print
