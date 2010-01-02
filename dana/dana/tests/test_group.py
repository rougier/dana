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
from dana.tests.tools import np_equal, np_almost_equal
from nose.tools import *

def test_group_1():
    ''' Check group creation '''
    G = dana.group((2,))
    assert (G.shape == (2,))

def test_group_2():
    ''' Check group creation '''
    G = dana.group((2,2))
    assert (G.shape == (2,2))

def test_group_3():
    ''' Check group creation '''
    G = dana.group((2,2,2))
    assert (G.shape == (2,2,2))

def test_ones_1():
    ''' Check group basic creation routine 'ones' '''
    G = dana.ones((2,))
    assert (np_equal(G.V, np.ones((2,))))

def test_ones_2():
    ''' Check group basic creation routine 'ones' '''
    G = dana.ones((2,2))
    assert (np_equal(G.V, np.ones((2,2))))

def test_ones_3 ():
    ''' Check group basic creation routine 'ones' '''
    G = dana.ones((2,2,2))
    assert (np_equal(G.V, np.ones((2,2,2))))

def test_zeros_1():
    ''' Check group basic creation routine 'zeros' '''
    G = dana.zeros((2,))
    assert (np_equal(G.V, np.zeros((2,))))

def test_zeros_2():
    ''' Check group basic creation routine 'zeros' '''
    G = dana.zeros((2,2))
    assert (np_equal(G.V, np.zeros((2,2))))

def test_zeros_3 ():
    ''' Check group basic creation routine 'zeros' '''
    G = dana.zeros((2,2,2))
    assert (np_equal(G.V, np.zeros((2,2,2))))

def test_empty_1():
    ''' Check group basic creation routine 'empty' '''
    G = dana.empty((2,))
    assert (G.shape == (2,))

def test_empty_2():
    ''' Check group basic creation routine 'empty' '''
    G = dana.empty((2,2))
    assert (G.shape == (2,2))

def test_empty_3 ():
    ''' Check group basic creation routine 'empty' '''
    G = dana.empty((2,2,2))
    assert (G.shape == (2,2,2))

def test_ones_like_1():
    ''' Check group basic creation routine 'ones_like' '''
    G = dana.ones_like(np.zeros((2,)))
    assert (np_equal(G.V, np.ones((2,))))

def test_ones_like_2():
    ''' Check group basic creation routine 'ones' '''
    G = dana.ones(np.zeros((2,2)))
    assert (np_equal(G.V, np.ones((2,2))))

def test_ones_like_3 ():
    ''' Check group basic creation routine 'ones' '''
    G = dana.ones(np.zeros((2,2,2)))
    assert (np_equal(G.V, np.ones((2,2,2))))

def test_zeros_like_1():
    ''' Check group basic creation routine 'zeros' '''
    G = dana.zeros(np.ones((2,)))
    assert (np_equal(G.V, np.zeros((2,))))

def test_zeros_like_2():
    ''' Check group basic creation routine 'zeros' '''
    G = dana.zeros(np.ones((2,2)))
    assert (np_equal(G.V, np.zeros((2,2))))

def test_zeros_like_3 ():
    ''' Check group basic creation routine 'zeros' '''
    G = dana.zeros(np.ones((2,2,2)))
    assert (np_equal(G.V, np.zeros((2,2,2))))

def test_empty_like_1():
    ''' Check group basic creation routine 'empty' '''
    G = dana.empty_like(np.empty((2,)))
    assert (G.shape == (2,))

def test_empty_like_2():
    ''' Check group basic creation routine 'empty' '''
    G = dana.empty_like(np.empty((2,2)))
    assert (G.shape == (2,2))

def test_empty_like_3 ():
    ''' Check group basic creation routine 'empty' '''
    G = dana.empty_like(np.empty((2,2,2)))
    assert (G.shape == (2,2,2))

def test_group_default():
    ''' Check group creation default parameters '''
    G = dana.group()
    assert (G.shape == ())
    assert ('V' in G.dtype.names)

def test_key_naming ():
    ''' Check group key naming '''
    G = dana.zeros((2,2), keys = ['U','W'])
    assert ('U' in G.dtype.names)
    assert ('W' in G.dtype.names)

def test_key_typing_1 ():
    ''' Check group key typing '''
    G = dana.zeros((2,2), keys = ['U','W'], dtype=int)
    assert (G.U.dtype == int)
    assert (G.W.dtype == int)

def test_key_typing_2 ():
    ''' Check group key typing '''
    G = dana.zeros((2,2), dtype = [('U',int), ('W',float)])
    assert (G.U.dtype == int)
    assert (G.W.dtype == float)

def test_key_access_1():
    ''' Check group key access '''
    G = dana.zeros((2,2))
    assert (np_equal(G.V,np.zeros((2,2))))

def test_key_access_2():
    ''' Check group key access '''
    G = dana.zeros((2,2))
    assert (np_equal(G['V'],np.zeros((2,2))))

def test_set_shape_1():
    ''' Check group set shape '''
    G = dana.zeros((2,2))
    G.shape = (4,1)
    assert (G.shape == (4,1))
    assert (G.V.shape == (4,1))

@raises(AttributeError)
def test_set_shape_2():
    ''' Check group set shape '''
    G = dana.zeros((2,2))
    G.V.shape = (4,1)

def test_reshape_1():
    ''' Check group reshape '''
    G = dana.zeros((2,2))
    assert (G.reshape((4,1)).shape == (4,1))
    assert (G.shape == (2,2))

def test_reshape_2():
    ''' Check group reshape '''
    G = dana.zeros((2,2))
    A = G.reshape((4,1))
    assert (A.shape == (4,1))

def test_parent():
    ''' Check group parent '''
    G = dana.zeros((2,2))
    assert (id(G.V.parent) == id(G))


def test_mask_1():
    ''' Check group mask '''
    G = dana.zeros((2,2))
    G.mask = False
    assert (np_equal(np.ones((2,2))*np.NaN,G.V))

def test_mask_2():
    ''' Check group mask '''
    G = dana.zeros((2,2))
    G.mask[0,0] = False
    Z = np.zeros((2,2))
    Z[0,0] = np.NaN
    assert (np_equal(Z,G.V))

def test_mask_3():
    ''' Check group mask '''
    G = dana.zeros((2,2))
    G.mask[0,0] = False
    G.mask[0,0] = True
    Z = np.zeros((2,2))
    assert (np_equal(Z,G.V))
