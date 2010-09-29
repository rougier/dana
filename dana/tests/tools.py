#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp

def np_almost_equal(A, B, epsilon=1e-10):
    ''' Assert two arrays are almost equal, even with NaN in them '''
    if sp.issparse(A):
        A = A.todense()
    if sp.issparse(B):
        B = B.todense()
    A_nan = np.isnan(A)
    B_nan = np.isnan(B)
    A_num = np.nan_to_num(A)
    B_num = np.nan_to_num(B)
    return np.all(A_nan==B_nan) and (np.abs(A_num-B_num)).sum() <= epsilon

def np_equal(A, B):
    ''' Assert two arrays are equal, even with NaN in them '''
    if sp.issparse(A): A = A.todense()
    if sp.issparse(B): B = B.todense()
    equal = np_almost_equal(A,B,epsilon = 1e-10)
    if not equal:
        print
        print A
        print 'is different from'
        print B
    return equal
