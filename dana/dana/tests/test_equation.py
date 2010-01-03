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

def test_no_equation():
    ''' Check equation when none '''
    G = dana.zeros((10,10))
    result = G.compute()
    assert result == []

def test_empty_equation():
    ''' Check equation when empty'''
    G = dana.zeros((10,10))
    G.dV = ''
    result = G.compute()
    assert result == []

def test_constant_equation():
    ''' Check equation with numerical constant'''
    G = dana.zeros((10,10))
    G.dV = '1.2345-V'
    result = G.compute()
    assert np_equal(G.V, np.ones((10,10))*1.2345)

def test_constant_access():
    ''' Check equation with symbolic constant'''
    G = dana.zeros((10,10))
    h = 1.2345
    G.dV = 'h-V'
    result = G.compute()
    assert np_equal(G.V, np.ones((10,10))*h)
    assert_almost_equal(result[0], 10*10*h)

def test_value_access():
    ''' Check equation value access'''
    G = dana.ones((10,10), keys=['U','V'])
    G.dU = '0'
    G.dV = 'U/2'
    result = G.compute()
    assert np_equal(G.V, np.ones((10,10))*1.5)
    assert_almost_equal(result[0],10*10*0.0)
    assert_almost_equal(result[1],10*10*0.5)

def test_cos_operator():
    ''' Check equation cos operator '''
    Z = np.random.random((10,10))
    G = dana.zeros((10,10))
    G.V = Z
    G.dV = 'cos(V)-V'
    result = G.compute()
    assert np_almost_equal(G.V,np.cos(Z))

def test_sin_operator():
    ''' Check equation sin operator '''
    Z = np.random.random((10,10))
    G = dana.zeros((10,10))
    G.V = Z
    G.dV = 'sin(V)-V'
    result = G.compute()
    assert np_almost_equal(G.V,np.sin(Z))


def test_exp_operator():
    ''' Check equation exp operator '''
    Z = np.random.random((10,10))
    G = dana.zeros((10,10))
    G.V = Z
    G.dV = 'exp(V)-V'
    result = G.compute()
    assert np_almost_equal(G.V,np.exp(Z))

def test_sqrt_operator():
    ''' Check equation sqrt operator '''
    Z = np.random.random((10,10))
    G = dana.zeros((10,10))
    G.V = Z
    G.dV = 'sqrt(V)-V'
    result = G.compute()
    assert np_almost_equal(G.V,np.sqrt(Z))


def test_equation_mask():
    ''' Check equation when there are masked elements '''
    Z = np.ones((10,10))
    G = dana.zeros((10,10))
    G.mask[0] = 0
    G.dV = '1-V'
    result = G.compute()
    Z[0] = np.nan
    assert np_equal(G.V, Z)

#     def test_equation_distance_computation (self):
#         ''' Check equation distance computation '''

#         n = 50
#         Z = np.random.random((n,n))
#         src = dana.group(Z)
#         dst = dana.zeros((n,n))
#         K = sp.identity(n*n, format='csr')
#         dst.connect(src, K, 'I-')
#         dst.dV = 'I'
#         dst.compute()
#         self.assert_ (self.almost_equal (dst.V, abs(Z-1)))


# # Test suite
# suite = unittest.TestLoader().loadTestsFromTestCase(equation)
# if __name__ == "__main__":
#     unittest.main()
