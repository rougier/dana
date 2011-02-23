#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import unittest
from numpy import *
import scipy.sparse as sp
from tools import np_equal
from dana import DenseConnection as Connection


class DenseOneDimensionTestCase(unittest.TestCase):
    def test_1(self):
        assert np_equal( Connection(ones(3), ones(3), ones(1)).output(),
                         ones(3))
    def test_2(self):
        assert np_equal( Connection(ones(3), ones(5), ones(1)).output(),
                         ones(5))
    def test_3(self):
        assert np_equal( Connection(ones(5), ones(3), ones(1)).output(),
                         ones(3))
    def test_4(self):
        assert np_equal( Connection(ones(3), ones(3), ones(3)).output(),
                         array([2,3,2]))
    def test_5(self):
        assert np_equal( Connection(ones(3), ones(5), ones(3)).output(),
                         array([2,2,3,2,2]))
    def test_6(self):
        assert np_equal( Connection(ones(5), ones(3), ones(3)).output(),
                         array([2,3,2]))
    def test_7(self):
        assert np_equal( Connection(ones(3), ones(3), array([1,NaN,1])).output(),
                         array([1,2,1]))
    def test_8(self):
        assert np_equal( Connection(ones(3), ones(3), array([NaN,NaN,NaN])).output(),
                         zeros(3))
    def test_9(self):
        assert np_equal( Connection(ones(3), ones(3), ones((3,3))).output(),
                         3*ones(3))
    def test_10(self):
        C = Connection(ones(3), ones(3), ones(1))
        assert np_equal(C[0], array([1, NaN, NaN]))
        assert np_equal(C[1], array([NaN, 1, NaN]))
        assert np_equal(C[2], array([NaN, NaN, 1]))

    def test_11(self):
        assert np_equal(Connection(ones(3), ones(3), ones(3), toric=True).output(),
                        ones(3)*3)

    def test_12(self):
        Z = ones(5)
        K = arange(5)
        C = Connection(Z,Z,K)
        assert np_equal(C[0], array([2,3,4,NaN,NaN]))

    def test_13(self):
        Z = ones(5)
        K = arange(5)
        C = Connection(Z,Z,K)
        assert np_equal(C[2],K)

    def test_14(self):
        Z = ones(5)
        K = arange(5)
        C = Connection(Z,Z,K)
        assert np_equal(C[4], array([NaN,NaN,0,1,2]))




class DenseTwoDimensionTestCase(unittest.TestCase):
    def test_1(self):
        assert np_equal( Connection(ones((3,3)), ones((3,3)), ones((1,1))).output(),
                         ones((3,3)))
    def test_2(self):
        assert np_equal( Connection(ones((3,3)), ones((5,5)), ones((1,1))).output(),
                         ones((5,5)))
    def test_3(self):
        assert np_equal( Connection(ones((5,5)), ones((3,3)), ones((1,1))).output(),
                         ones((3,3)))
    def test_4(self):
        assert np_equal( Connection(ones((3,3)), ones((3,3)), ones((3,3))).output(),
                         array([[4,6,4],
                                [6,9,6],
                                [4,6,4]]))
    def test_5(self):
        assert np_equal( Connection(ones((3,3)), ones((5,5)), ones((3,3))).output(),
                         array([[4,4,6,4,4],
                                [4,4,6,4,4],
                                [6,6,9,6,6],
                                [4,4,6,4,4],
                                [4,4,6,4,4]]))
    def test_6(self):
        assert np_equal( Connection(ones((5,5)), ones((3,3)), ones((3,3))).output(),
                         array([[4,6,4],
                                [6,9,6],
                                [4,6,4]]))

    def test_7(self):
        assert np_equal( Connection(ones((3,3)), ones((3,3)), array([[1, 1, 1],
                                                                     [1,NaN,1],
                                                                     [1, 1, 1]])).output(),
                         array([[3,5,3],
                                [5,8,5],
                                [3,5,3]]))
    def test_8(self):
        assert np_equal( Connection(ones((3,3)), ones((3,3)), ones((3,3))*NaN).output(),
                         zeros((3,3)) )

    def test_9(self):
        assert np_equal( Connection(ones((3,3)), ones((3,3)), ones((9,9))).output(),
                         9*ones((3,3)))

    def test_10(self):
        C = Connection(ones((3,3)), ones((3,3)), ones((1,1)))
        assert np_equal(C[1,1], array([[NaN, NaN, NaN],
                                       [NaN,  1,  NaN],
                                       [NaN, NaN, NaN]]))

    def test_11(self):
        assert np_equal(Connection(ones((3,3)), ones((3,3)), ones((3,3)), toric=True).output(),
                        ones((3,3))*9)

    def test_12(self):
        Z = ones((5,5))
        K = arange(5*5).reshape((5,5))
        C = Connection(Z,Z,K)
        assert np_equal(C[0,0],
                        array([[12, 13, 14,NaN,NaN],
                               [17, 18, 19,NaN,NaN],
                               [22, 23, 24,NaN,NaN],
                               [NaN,NaN,NaN,NaN,NaN],
                               [NaN,NaN,NaN,NaN,NaN]]))

    def test_13(self):
        Z = ones((5,5))
        K = arange(5*5).reshape((5,5))
        C = Connection(Z,Z,K)
        assert np_equal(C[0,4],
                        array([[NaN,NaN,10, 11, 12],
                               [NaN,NaN,15, 16, 17],
                               [NaN,NaN,20, 21, 22],
                               [NaN,NaN,NaN,NaN,NaN],
                               [NaN,NaN,NaN,NaN,NaN]]))

    def test_14(self):
        Z = ones((5,5))
        K = arange(5*5).reshape((5,5))
        C = Connection(Z,Z,K)
        assert np_equal(C[4,0],
                        array([[NaN,NaN,NaN,NaN,NaN],
                               [NaN,NaN,NaN,NaN,NaN],
                               [  2,  3,  4,NaN,NaN],
                               [  7,  8,  9,NaN,NaN],
                               [ 12, 13, 14,NaN,NaN]]))

    def test_15(self):
        Z = ones((5,5))
        K = arange(5*5).reshape((5,5))
        C = Connection(Z,Z,K)
        assert np_equal(C[4,4],
                        array([[NaN,NaN,NaN,NaN,NaN],
                               [NaN,NaN,NaN,NaN,NaN],
                               [NaN,NaN,  0,  1,  2],
                               [NaN,NaN,  5,  6,  7],
                               [NaN,NaN, 10, 11, 12]]))

    def test_16(self):
        Z = ones((5,5))
        K = arange(5*5).reshape((5,5))
        C = Connection(Z,Z,K)
        assert np_equal(C[2,2],K)
                        

if __name__ == "__main__":
    unittest.main()
