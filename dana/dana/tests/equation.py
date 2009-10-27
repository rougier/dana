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
import unittest
import numpy as np
import scipy.sparse as sp


class equation (unittest.TestCase):
    def setUp (self):
        pass
    
    def equal(self, A, B):
        ''' Assert two arrays are equal, even with NaN in them '''

        if sp.issparse(A): A = A.todense()
        if sp.issparse(B): B = B.todense()
        return self.almost_equal(A,B,epsilon = 0)

    def almost_equal(self, A, B, epsilon=1e-5):
        ''' Assert two arrays are almost equal, even with NaN in them '''

        if sp.issparse(A): A = A.todense()
        if sp.issparse(B): B = B.todense()
        A_nan = np.isnan(A)
        B_nan = np.isnan(B)
        A_num = np.nan_to_num(A)
        B_num = np.nan_to_num(B)
        return np.all(A_nan==B_nan) and (abs(A_num-B_num)).sum() <= epsilon

    def test_no_equation (self):
        ''' Check equation when none '''
        G = dana.zeros((10,10))
        result = G.compute()
        self.assertEqual (result, 0.0)

    def test_empty_equation (self):
        ''' Check equation when empty'''
        G = dana.zeros((10,10))
        G.dV = ''
        result = G.compute()
        self.assertEqual (result, 0.0)

    def test_constant_equation (self):
        ''' Check equation with numerical constant'''
        G = dana.zeros((10,10))
        G.dV = '1.2345'
        result = G.compute()
        self.assert_ (self.equal (G.V, np.ones((10,10))*1.2345))

#     def test_constant_access (self):
#         ''' Check equation with symbolic constant'''
#         G = dana.zeros((10,10))
#         G.constant['h'] = 1.2345
#         G.dV = 'h'
#         result = G.compute()
#         self.assert_ (self.equal (G.V, np.ones((10,10))*1.2345))

    def test_value_access (self):
        ''' Check equation value access'''
        G = dana.ones((10,10), keys=['V','U'])
        G.dV = 'V+U/2'
        result = G.compute()
        self.assert_ (self.equal (G.V, np.ones((10,10))*1.5))

    def test_cos_operator (self):
        ''' Check equation cos operator '''
        Z = np.random.random((10,10))
        G = dana.group(Z)
        G.dV = 'cos(V)'
        result = G.compute()
        self.assert_ (self.equal (G.V, np.cos(Z)))

    def test_sin_operator (self):
        ''' Check equation sin operator '''
        Z = np.random.random((10,10))
        G = dana.group(Z)
        G.dV = 'sin(V)'
        result = G.compute()
        self.assert_ (self.equal (G.V, np.sin(Z)))

    def test_exp_operator (self):
        ''' Check equation exp operator '''
        Z = np.random.random((10,10))
        G = dana.group(Z)
        G.dV = 'exp(V)'
        result = G.compute()
        self.assert_ (self.equal (G.V, np.exp(Z)))

    def test_sqrt_operator (self):
        ''' Check equation sqrt operator '''
        Z = np.random.random((10,10))
        G = dana.group(Z)
        G.dV = 'sqrt(V)'
        result = G.compute()
        self.assert_ (self.equal (G.V, np.sqrt(Z)))

    def test_equation_mask (self):
        ''' Check equation when there are masked elements '''
        Z = np.ones((10,10))
        G = dana.group(Z)
        G.mask[0] = 0
        G.dV = '1'
        result = G.compute()
        Z[0] = 0
        self.assert_ (self.equal (G.V, Z))


    def test_equation_distance_computation (self):
        ''' Check equation distance computation '''

        n = 50
        Z = np.random.random((n,n))
        src = dana.group(Z)
        dst = dana.zeros((n,n))
        K = sp.identity(n*n, format='csr')
        dst.connect(src, K, 'I-')
        dst.dV = 'I'
        dst.compute()
        self.assert_ (self.almost_equal (dst.V, abs(Z-1)))


# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(equation)
if __name__ == "__main__":
    unittest.main()
