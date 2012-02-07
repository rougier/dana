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
from dana import Group, zeros, ones
from dana import DenseConnection, SparseConnection


class LearningDenseOneDimensionTestCase(unittest.TestCase):
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
                             equation = 'dW/dt = post.I')
        dst.run(dt=0.1)
        assert np_equal(C.weights, np.identity(3)*1.1)


class LearningSparseOneDimensionTestCase(unittest.TestCase):
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
        C.evaluate(dt=0.1)
        assert np_equal(C.weights, np.array([[0,1,0],
                                             [1,0,1],
                                             [0,1,0]])*1.1)

    def test_4(self):
        src = np.ones((3,))
        dst = zeros((3,) , 'V=I; I')
        kernel = np.ones(1)
        C = SparseConnection(src, dst('I'), kernel,
                             equation = 'dW/dt = post.I')
        dst.run(dt=0.1)
        assert np_equal(C.weights, np.identity(3)*1.1)


if __name__ == "__main__":
    unittest.main()
