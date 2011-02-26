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
