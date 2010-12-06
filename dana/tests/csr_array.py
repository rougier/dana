#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
import unittest
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from dana import *


def np_almost_equal(A, B, epsilon=1e-10):
    if issparse(A):
        A = A.todense()
    if issparse(B):
        B = B.todense()
    A_nan = np.isnan(A)
    B_nan = np.isnan(B)
    A_num = np.nan_to_num(A)
    B_num = np.nan_to_num(B)
    return np.all(A_nan==B_nan) and (abs(A_num-B_num)).sum() <= epsilon

def np_equal(A, B):
    if issparse(A):
        A = A.todense()
    if issparse(B):
        B = B.todense()
    return np_almost_equal(A,B,epsilon = 0)


class TestArray(unittest.TestCase):

    def test_add(self):
        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        assert (np_almost_equal(A+1,As+1))
        assert (np_almost_equal(1+A,1+As))
        assert (np_almost_equal(A+B,As+B))
        assert (np_almost_equal(B+A,B+As))
        assert (np_almost_equal(A+C,As+C))
        assert (np_almost_equal(C+A,C+As))
        assert (np_almost_equal(A+D,As+D))
        assert (np_almost_equal(D+A,D+As))

    def test_sub(self):
        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        assert (np_almost_equal(A-1,As-1))
        assert (np_almost_equal(1-A,1-As))
        assert (np_almost_equal(A-B,As-B))
        assert (np_almost_equal(B-A,B-As))
        assert (np_almost_equal(A-C,As-C))
        assert (np_almost_equal(C-A,C-As))
        assert (np_almost_equal(A-D,As-D))
        assert (np_almost_equal(D-A,D-As))

    def test_mul(self):
        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        assert (np_almost_equal(A*1,As*1))
        assert (np_almost_equal(1*A,1*As))
        assert (np_almost_equal(A*B,As*B))
        assert (np_almost_equal(B*A,B*As))
        assert (np_almost_equal(A*C,As*C))
        assert (np_almost_equal(C*A,C*As))
        assert (np_almost_equal(A*D,As*D))
        assert (np_almost_equal(D*A,D*As))

    def test_div(self):
        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        assert (np_almost_equal(A/1,As/1))
        assert (np_almost_equal(1/A,1/As))
        assert (np_almost_equal(A/B,As/B))
        assert (np_almost_equal(B/A,B/As))
        assert (np_almost_equal(A/C,As/C))
        assert (np_almost_equal(C/A,C/As))
        assert (np_almost_equal(A/D,As/D))
        assert (np_almost_equal(D/A,D/As))

    def test_dot(self):
        A = np.random.random((5,10))+1
        B = np.random.random((10,1))
        As = csr_array(A)
        Bs = csr_array(B)
        assert (np_almost_equal(np.dot(A,B), dot(As,B)))

    def test_sum(self):
        A = np.random.random((5,10))+1
        As = csr_array(A)
        assert (np_almost_equal(A.sum(), As.sum()))
        assert (np_almost_equal(A.sum(axis=0), As.sum(axis=0)))

    def test_misc(self):
        A = np.random.random((5,10))+1
        As = csr_array(A)
        assert (np_almost_equal(np.cos(A), np.cos(As)))
        assert (np_almost_equal(np.sin(A), np.sin(As)))
        assert (np_almost_equal(np.exp(A), np.exp(As)))
        assert (np_almost_equal(np.sqrt(A),np.sqrt(As)))

    def test_mask(self):
        A = np.zeros((5,5))
        A[0,0] = 1
        As = csr_array(A)
        A[1,0] = 1
        As += A
        assert (As[0,0] == 2)
        assert (As[1,0] == 0)


if __name__ == '__main__':
    unittest.main()
