#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import unittest
import numpy as np
import scipy.sparse as sp
from tools import np_equal
from dana import Group, zeros, ones
from dana import DenseConnection, SparseConnection


class DenseOneDimensionTestCase(unittest.TestCase):
    def test_1(self):
        kernel = np.ones(1)
        C = DenseConnection(np.ones(3), np.ones(3), kernel,
                             equation = 'dW/dt = 1')
        C.evaluate(dt=.1)
        assert np_equal(C.weights, np.identity(3)*1.1)

    def test_2(self):
        kernel = np.ones(3)*np.NaN
        C = DenseConnection(np.ones(3), np.ones(3), kernel,
                             equation = 'dW/dt = 1')
        C.evaluate(dt=.1)
        assert np_equal(C.weights, np.zeros((3,3)))

    def test_3(self):
        kernel = np.ones(3)
        kernel[1] = np.NaN
        C = DenseConnection(np.ones(3), np.ones(3), kernel,
                            equation = 'dW/dt = 1')
        C.evaluate(dt=.1)
        assert np_equal(C.weights, np.array([[0,1,0],
                                             [1,0,1],
                                             [0,1,0]])*1.1)

    def test_4(self):
        src = np.ones((3,))
        dst = zeros((3,) , 'V=I; I')
        kernel = np.ones(1)
        C = DenseConnection(src, dst('I'), kernel,
                             equation = 'dW/dt = I')
        dst.run(t=0.1, dt=0.1)
        assert np_equal(C.weights, np.identity(3)*1.1)


class SparseOneDimensionTestCase(unittest.TestCase):
    def test_1(self):
        kernel = np.ones(1)
        C = SparseConnection(np.ones(3), np.ones(3), kernel,
                             equation = 'dW/dt = 1')
        C.evaluate(dt=.1)
        assert np_equal(C.weights, np.identity(3)*1.1)

    def test_2(self):
        kernel = np.ones(3)*np.NaN
        C = SparseConnection(np.ones(3), np.ones(3), kernel,
                             equation = 'dW/dt = 1')
        C.evaluate(dt=.1)
        assert np_equal(C.weights, np.zeros((3,3)))

    def test_3(self):
        kernel = np.ones(3)
        kernel[1] = np.NaN
        C = SparseConnection(np.ones(3), np.ones(3), kernel,
                             equation = 'dW/dt = 1')
        C.evaluate(dt=.1)
        assert np_equal(C.weights, np.array([[0,1,0],
                                             [1,0,1],
                                             [0,1,0]])*1.1)

    def test_4(self):
        src = np.ones((3,))
        dst = zeros((3,) , 'V=I; I')
        kernel = np.ones(1)
        C = SparseConnection(src, dst('I'), kernel,
                             equation = 'dW/dt = I')
        dst.run(t=0.1, dt=0.1)
        assert np_equal(C.weights, np.identity(3)*1.1)



if __name__ == "__main__":
    unittest.main()
