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
