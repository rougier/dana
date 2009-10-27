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
import unittest
import numpy as np
import scipy.sparse as sp
from dana.array import csr_array, dot

class array (unittest.TestCase):
    def setUp (self):
        pass
    
    def equal(self, A, B):
        ''' Assert two arrays are equal, even with NaN in them '''

        if sp.issparse(A): A = A.todense()
        if sp.issparse(B): B = B.todense()
        return self.almost_equal(A,B,epsilon = 0)

    def almost_equal(self, A, B, epsilon=1e-10):
        ''' Assert two arrays are almost equal, even with NaN in them '''

        if sp.issparse(A): A = A.todense()
        if sp.issparse(B): B = B.todense()
        A_nan = np.isnan(A)
        B_nan = np.isnan(B)
        A_num = np.nan_to_num(A)
        B_num = np.nan_to_num(B)
        return np.all(A_nan==B_nan) and (abs(A_num-B_num)).sum() <= epsilon

    def test_add (self):
        ''' Check array addition '''        

        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        self.assert_ (self.almost_equal(A+1,As+1))
        self.assert_ (self.almost_equal(1+A,1+As))
        self.assert_ (self.almost_equal(A+B,As+B))
        self.assert_ (self.almost_equal(B+A,B+As))
        self.assert_ (self.almost_equal(A+C,As+C))
        self.assert_ (self.almost_equal(C+A,C+As))
        self.assert_ (self.almost_equal(A+D,As+D))
        self.assert_ (self.almost_equal(D+A,D+As))

    def test_sub (self):
        ''' Check array subtract '''        

        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        self.assert_ (self.almost_equal(A-1,As-1))
        self.assert_ (self.almost_equal(1-A,1-As))
        self.assert_ (self.almost_equal(A-B,As-B))
        self.assert_ (self.almost_equal(B-A,B-As))
        self.assert_ (self.almost_equal(A-C,As-C))
        self.assert_ (self.almost_equal(C-A,C-As))
        self.assert_ (self.almost_equal(A-D,As-D))
        self.assert_ (self.almost_equal(D-A,D-As))

    def test_mul (self):
        ''' Check array multiplication '''        

        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        self.assert_ (self.almost_equal(A*1,As*1))
        self.assert_ (self.almost_equal(1*A,1*As))
        self.assert_ (self.almost_equal(A*B,As*B))
        self.assert_ (self.almost_equal(B*A,B*As))
        self.assert_ (self.almost_equal(A*C,As*C))
        self.assert_ (self.almost_equal(C*A,C*As))
        self.assert_ (self.almost_equal(A*D,As*D))
        self.assert_ (self.almost_equal(D*A,D*As))

    def test_div (self):
        ''' Check array division '''        

        A = np.random.random((5,10))+1
        B = np.random.random((5,10))+1
        C = np.random.random((5,1))
        D = np.random.random((10,))
        As = csr_array(A)
        Bs = csr_array(B)
        self.assert_ (self.almost_equal(A/1,As/1))
        self.assert_ (self.almost_equal(1/A,1/As))
        self.assert_ (self.almost_equal(A/B,As/B))
        self.assert_ (self.almost_equal(B/A,B/As))
        self.assert_ (self.almost_equal(A/C,As/C))
        self.assert_ (self.almost_equal(C/A,C/As))
        self.assert_ (self.almost_equal(A/D,As/D))
        self.assert_ (self.almost_equal(D/A,D/As))


    def test_dot (self):
        ''' Check array dot product '''        

        A = np.random.random((5,10))+1
        B = np.random.random((10,1))
        As = csr_array(A)
        Bs = csr_array(B)
        self.assert_ (self.almost_equal(np.dot(A,B), dot(As,B)))

    def test_sum (self):
        ''' Check array sum '''        

        A = np.random.random((5,10))+1
        As = csr_array(A)
        self.assert_ (self.almost_equal(A.sum(), As.sum()))
        self.assert_ (self.almost_equal(A.sum(axis=0), As.sum(axis=0)))
        #self.assert_ (self.almost_equal(A.sum(axis=1), As.sum(axis=1)))


    def test_misc (self):
        ''' Check array miscellaneous operations '''        

        A = np.random.random((5,10))+1
        As = csr_array(A)
        self.assert_ (self.almost_equal(np.cos(A), np.cos(As)))
        self.assert_ (self.almost_equal(np.sin(A), np.sin(As)))
        self.assert_ (self.almost_equal(np.exp(A), np.exp(As)))
        self.assert_ (self.almost_equal(np.sqrt(A),np.sqrt(As)))


# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(array)
if __name__ == "__main__":
    unittest.main()
