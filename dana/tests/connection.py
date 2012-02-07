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
import unittest
import numpy as np
import scipy.sparse as sp
from tools import np_equal
from dana import ConnectionError
from dana import SparseConnection, DenseConnection, SharedConnection
from scipy.ndimage.filters import convolve, convolve1d

class ConnectionOneDimensionTestCase(unittest.TestCase):

    def test_1(self):
        n = 9
        Z = np.random.random(n)
        K = np.random.random(n//2)
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SparseConnection(Z,Z,K,toric=True).output()
        Z3 = SharedConnection(Z,Z,K,toric=True).output()
        Z4 = convolve(Z, K[::-1], mode='wrap')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_2(self):
        n = 9
        Z = np.random.random(n)
        K = np.random.random(n)
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='wrap')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_3(self):
        n = 9
        Z = np.random.random(n)
        K = np.random.random(2*n)
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        K_ = K[5:14]
        Z4 = convolve(Z, K_[::-1], mode='wrap')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_4(self):
        n = 10
        Z = np.random.random(n)
        K = np.random.random(n//2)
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='wrap')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_5(self):
        n = 10
        Z = np.random.random(n)
        K = np.random.random(n)
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='wrap')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_6(self):
        n = 10
        Z = np.random.random(n)
        K = np.random.random(2*n)
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        K_ = K[5:15]
        Z4 = convolve(Z, K_[::-1], mode='wrap')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)


    def test_7(self):
        n = 9
        Z = np.random.random(n)
        K = np.random.random(n//2)
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='constant')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_8(self):
        n = 9
        Z = np.random.random(n)
        K = np.random.random(n)
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='constant')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_9(self):
        n = 9
        Z = np.random.random(n)
        K = np.random.random(2*n)
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='constant')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_10(self):
        n = 10
        Z = np.random.random(n)
        K = np.random.random(n//2)
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='constant')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_11(self):
        n = 10
        Z = np.random.random(n)
        K = np.random.random(n)
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='constant')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)

    def test_12(self):
        n = 10
        Z = np.random.random(n)
        K = np.random.random(2*n)
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z4 = convolve(Z, K[::-1], mode='constant')
        assert np_equal(Z1,Z4)
        assert np_equal(Z2,Z4)
        assert np_equal(Z3,Z4)


class ConnectionTwoDimensionTestCase(unittest.TestCase):

    def test_1(self):
        n = 9
        Z = np.random.random((n,n))
        K = np.random.random((n//2,n//2))
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SparseConnection(Z,Z,K,toric=True).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='wrap')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_2(self):
        n = 9
        Z = np.random.random((n,n))
        K = np.random.random((n,n))
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SparseConnection(Z,Z,K,toric=True).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='wrap')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_3(self):
        n = 9
        Z = np.random.random((n,n))
        K = np.random.random((2*n,2*n))
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SparseConnection(Z,Z,K,toric=True).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        K_ = K[5:14,5:14]
        Z5 = convolve(Z, K_[::-1,::-1], mode='wrap')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_4(self):
        n = 10
        Z = np.random.random((n,n))
        K = np.random.random((n//2,n//2))
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SparseConnection(Z,Z,K,toric=True).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='wrap')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_5(self):
        n = 10
        Z = np.random.random((n,n))
        K = np.random.random((n,n))
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SparseConnection(Z,Z,K,toric=True).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='wrap')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_6(self):
        n = 10
        Z = np.random.random((n,n))
        K = np.random.random((2*n,2*n))
        Z1 = DenseConnection(Z,Z,K,toric=True).output()
        Z2 = SparseConnection(Z,Z,K,toric=True).output()
        Z3 = SharedConnection(Z,Z,K,toric=True,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=True,fft=True).output()
        K_ = K[5:15,5:15]
        Z5 = convolve(Z, K_[::-1,::-1], mode='wrap')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)


    def test_7(self):
        n = 9
        Z = np.random.random((n,n))
        K = np.random.random((n//2,n//2))
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SparseConnection(Z,Z,K,toric=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='constant')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_8(self):
        n = 9
        Z = np.random.random((n,n))
        K = np.random.random((n,n))
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SparseConnection(Z,Z,K,toric=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='constant')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_9(self):
        n = 9
        Z = np.random.random((n,n))
        K = np.random.random((2*n,2*n))
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SparseConnection(Z,Z,K,toric=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='constant')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_10(self):
        n = 10
        Z = np.random.random((n,n))
        K = np.random.random((n//2,n//2))
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SparseConnection(Z,Z,K,toric=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='constant')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_11(self):
        n = 10
        Z = np.random.random((n,n))
        K = np.random.random((n,n))
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SparseConnection(Z,Z,K,toric=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='constant')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)

    def test_12(self):
        n = 10
        Z = np.random.random((n,n))
        K = np.random.random((2*n,2*n))
        Z1 = DenseConnection(Z,Z,K,toric=False).output()
        Z2 = SparseConnection(Z,Z,K,toric=False).output()
        Z3 = SharedConnection(Z,Z,K,toric=False,fft=False).output()
        Z4 = SharedConnection(Z,Z,K,toric=False,fft=True).output()
        Z5 = convolve(Z, K[::-1,::-1], mode='constant')
        assert np_equal(Z1,Z5)
        assert np_equal(Z2,Z5)
        assert np_equal(Z3,Z5)
        assert np_equal(Z4,Z5)


if __name__ == "__main__":
    unittest.main()
