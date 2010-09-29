#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE
import dana
import numpy as np
from dana.tests.tools import *
from nose.tools import *


def test_one_to_one_dense_1():
    ''' Check link computation, one to one, dense '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=False, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_dense_2():
    ''' Check link computation, one to one, dense '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_dense_3():
    ''' Check link computation, one to one, dense '''

    G1 = dana.group(np.random.random((5,5)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V[::2,::2])

def test_one_to_one_dense_4():
    ''' Check link, one to one, dense '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=False, shared=False)
    W = G1.I[0]
    Z = np.ones((10,))*np.NaN
    Z[0] = 1
    assert np_almost_equal(W,Z)

def test_one_to_one_dense_5():
    ''' Check link, one to one, dense '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=False)
    W = G1.I[0,0]
    Z = np.ones((10,10))*np.NaN
    Z[0,0] = 1
    assert np_almost_equal(W,Z)

def test_one_to_one_dense_6():
    ''' Check link computation, one to one, dense, masked '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=False, shared=False)
    G1.mask[0] = False
    G1.dV = 'I-V'
    G1.compute()
    G2.V[0] = 0
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_dense_7():
    ''' Check link computation, one to one, dense, masked '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=False)
    G1.mask[0,0] = False
    G1.dV = 'I-V'
    G1.compute()
    G2.V[0,0] = 0
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_shared_1():
    ''' Check link computation, one to one, shared '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=False, shared=True)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_shared_2():
    ''' Check link computation, one to one, shared '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=True)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_shared_3():
    ''' Check link computation, one to one, shared '''

    G1 = dana.group(np.random.random((5,5)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=True)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V[::2,::2])

def test_one_to_one_shared_4():
    ''' Check link, one to one, shared '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=False, shared=True)
    W = G1.I[0]
    Z = np.ones((10,))*np.NaN
    Z[0] = 1
    assert np_almost_equal(W,Z)

def test_one_to_one_shared_5():
    ''' Check link, one to one, shared '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=True)
    W = G1.I[0,0]
    Z = np.ones((10,10))*np.NaN
    Z[0,0] = 1
    assert np_almost_equal(W,Z)

def test_one_to_one_shared_6():
    ''' Check link, one to one, shared '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=False, shared=True)
    G1.mask[0] = False
    G1.dV = 'I-V'
    G1.compute()
    G2.V[0] = 0
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_shared_7():
    ''' Check link, one to one, shared '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=False, shared=True)
    G1.mask[0,0] = False
    G1.dV = 'I-V'
    G1.compute()
    G2.V[0,0] = 0
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_sparse_1():
    ''' Check link computation, one to one, dense '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=True, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_sparse_2():
    ''' Check link computation, one to one, dense '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=True, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_sparse_3():
    ''' Check link computation, one to one, dense '''

    G1 = dana.group(np.random.random((5,5)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=True, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,G2.V[::2,::2])

def test_one_to_one_sparsed_4():
    ''' Check link, one to one, sparse '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=True, shared=False)
    W = G1.I[0]
    Z = np.ones((10,))*np.NaN
    Z[0] = 1
    assert np_almost_equal(W,Z)

def test_one_to_one_sparse_5():
    ''' Check link, one to one, sparse '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=True, shared=False)
    W = G1.I[0,0]
    Z = np.ones((10,10))*np.NaN
    Z[0,0] = 1
    assert np_almost_equal(W,Z)

def test_one_to_one_sparse_6():
    ''' Check link, one to one, sparse '''

    G1 = dana.group(np.random.random((10,)))
    G2 = dana.group(np.random.random((10,)))
    G1.connect(G2, np.ones((1,)), 'I', sparse=True, shared=False)
    G1.mask[0] = False
    G1.dV = 'I-V'
    G1.compute()
    G2.V[0] = 0
    assert np_almost_equal(G1.V,G2.V)

def test_one_to_one_sparse_7():
    ''' Check link, one to one, sparse '''

    G1 = dana.group(np.random.random((10,10)))
    G2 = dana.group(np.random.random((10,10)))
    G1.connect(G2, np.ones((1,1)), 'I', sparse=True, shared=False)
    G1.mask[0,0] = False
    G1.dV = 'I-V'
    G1.compute()
    G2.V[0,0] = 0
    assert np_almost_equal(G1.V,G2.V)

def test_weighted_sum_dense_1():
    ''' Check link weighted sum computation, dense '''

    G1 = dana.zeros((5,))
    G2 = dana.ones((5,))
    G1.connect(G2,np.ones((5,)), 'I', sparse=False, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([3,4,5,4,3]))

def test_weighted_sum_dense_2():
    ''' Check link weighted sum computation, dense '''

    G1 = dana.zeros((3,3))
    G2 = dana.ones((3,3))
    G1.connect(G2,np.ones((3,3)), 'I', sparse=False, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([[4,6,4],
                                           [6,9,6],
                                           [4,6,4]]))

def test_weighted_sum_dense_3():
    ''' Check link weighted sum computation, dense, masked '''

    G1 = dana.zeros((3,3))
    G2 = dana.ones((3,3))
    G1.connect(G2,np.ones((3,3)), 'I', sparse=False, shared=False)
    G2.mask[1,1] = False
    G1.dV = 'I-V'
    G1.compute()
    print G1.V
    assert np_almost_equal(G1.V, np.array([[3,5,3],
                                           [5,8,5],
                                           [3,5,3]]))

def test_weighted_sum_shared_1():
    ''' Check link weighted sum computation, shared '''

    G1 = dana.zeros((5,))
    G2 = dana.ones((5,))
    G1.connect(G2,np.ones((5,)), 'I', sparse=False, shared=True)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([3,4,5,4,3]))

def test_weighted_sum_shared_2():
    ''' Check link weighted sum computation, shared '''

    G1 = dana.zeros((3,3))
    G2 = dana.ones((3,3))
    G1.connect(G2,np.ones((3,3)), 'I', sparse=False, shared=True)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([[4,6,4],
                                           [6,9,6],
                                           [4,6,4]]))

def test_weighted_sum_shared_3():
    ''' Check link weighted sum computation, shared, masked '''

    G1 = dana.zeros((3,3))
    G2 = dana.ones((3,3))
    G1.connect(G2,np.ones((3,3)), 'I', sparse=False, shared=True)
    G2.mask[1,1] = False
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([[3,5,3],
                                           [5,8,5],
                                           [3,5,3]]))

def test_weighted_sum_sparse_1():
    ''' Check link weighted sum computation, sparse '''

    G1 = dana.zeros((5,))
    G2 = dana.ones((5,))
    G1.connect(G2,np.ones((5,)), 'I', sparse=True, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([3,4,5,4,3]))

def test_weighted_sum_sparse_2():
    ''' Check link weighted sum computation, sparse '''

    G1 = dana.zeros((3,3))
    G2 = dana.ones((3,3))
    G1.connect(G2,np.ones((3,3)), 'I', sparse=True, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([[4,6,4],
                                           [6,9,6],
                                           [4,6,4]]))

def test_weighted_sum_sparse_3():
    ''' Check link weighted sum computation, sparse, masked '''

    G1 = dana.zeros((3,3))
    G2 = dana.ones((3,3))
    G1.connect(G2,np.ones((3,3)), 'I', sparse=True, shared=False)
    G2.mask[1,1] = False
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V, np.array([[3,5,3],
                                           [5,8,5],
                                           [3,5,3]]))

def test_distance_dense_1():
    ''' Check link distance computation, dense '''

    G1 = dana.zeros((5,))
    G2 = dana.group(np.random.random((5,)))
    G1.connect(G2, np.ones((1,)), 'I-', sparse=False, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,1-G2.V)

def test_distance_dense_2():
    ''' Check link distance computation, dense '''

    G1 = dana.zeros((5,5))
    G2 = dana.group(np.random.random((5,5)))
    G1.connect(G2, np.ones((1,1)), 'I-', sparse=False, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,1-G2.V)

@raises(ValueError)
def test_distance_shared_1():
    ''' Check link distance computation, shared '''

    G1 = dana.zeros((5,))
    G2 = dana.group(np.random.random((5,)))
    G1.connect(G2, np.ones((1,)), 'I-', sparse=False, shared=True)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,1-G2.V)

@raises(ValueError)
def test_distance_shared_2():
    ''' Check link distance computation, dense '''

    G1 = dana.zeros((5,5))
    G2 = dana.group(np.random.random((5,5)))
    G1.connect(G2, np.ones((1,1)), 'I-', sparse=False, shared=True)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,1-G2.V)

def test_distance_sparse_1():
    ''' Check link distance computation, sparse '''

    G1 = dana.zeros((5,))
    G2 = dana.group(np.random.random((5,)))
    G1.connect(G2, np.ones((1,)), 'I-', sparse=True, shared=False)
    G1.dV = 'I-V'
    G1.compute()
    assert np_almost_equal(G1.V,1-G2.V)

def test_distance_sparse_2():
    ''' Check link distance computation, sparse '''

    G1 = dana.zeros((5,5))
    G2 = dana.group(np.random.random((5,5)))
    G1.connect(G2, np.ones((1,1)), 'I-', sparse=True, shared=False)
    G1.dV = 'I'
    G1.compute()
    assert np_almost_equal(G1.V,1-G2.V)

#     def test_distance_mask (self):
#         ''' Check link distance computation with mask'''

#         G1 = dana.ones((5,))
#         G2 = dana.zeros((5,))
#         G1.connect(G2.V,np.ones((1,)), 'I-')
#         G1.dV = 'I'
#         G1.compute()
#         self.assert_ (self.almost_equal (G1.V, np.array([1,1,1,1,1])))

