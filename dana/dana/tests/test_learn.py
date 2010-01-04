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
    ''' Check learning, one to one, dense '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=False, shared=False)
    G1.V *= 2.5
    G2.V *= 1.1
    G1.dI = 'pre.V-W'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10)*1.1)

def test_one_to_one_dense_2():
    ''' Check learning, one to one, dense '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=False, shared=False)
    G1.V *= 2.5
    G2.V *= 1.1
    G1.dI = 'post.V-W'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10)*2.5)

def test_one_to_one_dense_3():
    ''' Check learning, one to one, dense '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=False, shared=False)
    G1.V *= 2.5
    G2.V *= 1.1
    G1.dI = 'post.V*pre.V-W'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10)*2.75)

def test_one_to_one_dense_4():
    ''' Check learning, one to one, dense '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=False, shared=False)
    G1.dI = '-W'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.zeros((10,)))

def test_one_to_one_dense_5():
    ''' Check learning, one to one, dense '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=False, shared=False)
    G1.dI = '0'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10))


def test_one_to_one_sparse_1():
    ''' Check learning, one to one, sparse '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=True, shared=False)
    G1.V *= 2.5
    G2.V *= 1.1
    G1.dI = '-W+pre.V'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10)*1.1)

def test_one_to_one_sparse_2():
    ''' Check learning, one to one, sparse '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=True, shared=False)
    G1.V *= 2.5
    G2.V *= 1.1
    G1.dI = '-W+post.V'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10)*2.5)

def test_one_to_one_sparse_3():
    ''' Check learning, one to one, sparse '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=True, shared=False)
    G1.V *= 2.5
    G2.V *= 1.1
    G1.dI = '-W+post.V*pre.V'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10)*2.75)

def test_one_to_one_sparse_4():
    ''' Check learning, one to one, sparse '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=True, shared=False)
    G1.dI = '-W'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.zeros((10,10)))

def test_one_to_one_sparse_5():
    ''' Check learning, one to one, sparse '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=True, shared=False)
    G1.dI = '0'
    G1.learn()
    assert np_almost_equal(G1.I.kernel,np.eye(10))

@raises(ValueError)
def test_one_to_one_shared():
    ''' Check learning, one to one, shared '''

    G1 = dana.ones((10,))
    G2 = dana.ones((10,))
    G1.connect(G2.V, np.ones((1,)), 'I', sparse=False, shared=True)
    G1.dI = '0'
