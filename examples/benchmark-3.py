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
This script benchmarks connections depending on kernel size and sparsity
'''
import time
from dana import *


def test(shape, K, n):
    src, tgt = np.ones(shape), np.ones(shape)

    C = DenseConnection(src,tgt, K)
    t0 = time.clock()
    for i in range(n):
        C.propagate()
    print '    Dense Connection: ', time.clock()-t0

    C = SharedConnection(src,tgt, K)
    t0 = time.clock()
    for i in range(n):
        C.propagate()
    print '    Shared Connection:', time.clock()-t0

    C = SparseConnection(src,tgt, K)
    t0 = time.clock()
    for i in range(n):
        C.propagate()
    print '    Sparse Connection:', time.clock()-t0
    print

p = 1

for s in [0.1, 0.25, 0.5]:
    for n in [10,25,50]:
        p = np.maximum(1,int(s*n))
        print 'n=%d, K separable, sparsity = %d%%' % (n, s*100)
        test((n,n), np.ones((p,p)), 1000)

for s in [0.1, 0.25, 0.5]:
    for n in [10,25,50]:
        p = np.maximum(1,int(s*n))
        print 'n=%d, K non separable, sparsity = %d%%' % (n, s*100)
        test((n,n), np.random.random((p,p)), 1000)
