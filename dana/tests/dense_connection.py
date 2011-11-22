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
# by CEA, CNRS and INRIA at the following URL http://www.cecill.info.
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

    def test_11(self):
        assert np_equal(Connection(ones(3), ones(3), ones(3), toric=True).output(),
                        ones(3)*3)

    def test_11_bis(self):
        assert np_equal(Connection(ones(3), ones(3), ones(20), toric=True).output(),
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

    def test_11_bis(self):
        assert np_equal(Connection(ones((3,3)), ones((3,3)), ones((20,20)), toric=True).output(),
                        ones((3,3))*9)

    def test_11_ter(self):
        assert np_equal(Connection(ones((3,3)), ones((3,3)), ones((1,20)), toric=True).output(),
                        ones((3,3))*3)

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
