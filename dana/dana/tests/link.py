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
import dana
from dana.link import link

class link (unittest.TestCase):
    def setUp (self):
        pass

    def equal(self, A, B):
        ''' Assert two arrays are equal, even with NaN in them '''
        return self.almost_equal(A,B,epsilon = 0)

    def almost_equal(self, A, B, epsilon=1e-5):
        ''' Assert two arrays are almost equal, even with NaN in them '''

        A_nan = np.isnan(A)
        B_nan = np.isnan(B)
        A_num = np.nan_to_num(A)
        B_num = np.nan_to_num(B)
        return np.all(A_nan==B_nan) and (abs(A_num-B_num)).sum() <= epsilon


    def test_one_to_one (self):
        ''' Check link computation, one to one '''

        G1 = dana.group(np.random.random((3,3)))
        G2 = dana.group(np.random.random((3,3)))
        G1.connect(G2.V, np.ones((1,1)), 'I', sparse=True,  shared=False)
        G1.connect(G2.V, np.ones((1,1)), 'J', sparse=False, shared=True)
        G1.connect(G2.V, np.ones((1,1)), 'K', sparse=False, shared=False)

        G1.dV = 'I'
        G1.compute()
        self.assert_ (self.equal(G1.V,G2.V))
        G1.dV = 'J'
        G1.compute()
        self.assert_ (self.equal(G1.V,G2.V))
        G1.dV = 'K'
        G1.compute()
        self.assert_ (self.equal(G1.V,G2.V))


    def test_one_to_one_shared (self):
        ''' Check link, one to one, shared kernel '''

        G1 = dana.group(np.random.random((3,3)))
        G2 = dana.group(np.random.random((3,3)))
        R = np.ones((3,3))*np.NaN
        R[0,0] = 1
        G1.connect(G2.V, np.ones((1,1)), 'I', shared=True)
        self.assert_ (self.equal(G1.I[0,0],R))


    def test_one_to_one_sparse (self):
        ''' Check link, one to one, sparse kernel'''

        G1 = dana.group(np.random.random((3,3)))
        G2 = dana.group(np.random.random((3,3)))
        R = np.ones((3,3))*np.NaN
        R[0,0] = 1
        G1.connect(G2.V, np.ones((1,1)), 'I', sparse=True)
        self.assert_ (self.equal(G1.I[0,0],R))


    def test_one_to_one_dense (self):
        ''' Check link, one to one, dense kernel '''

        G1 = dana.group(np.random.random((3,3)))
        G2 = dana.group(np.random.random((3,3)))
        R = np.ones((3,3))*np.NaN
        R[0,0] = 1
        G1.connect(G2.V, np.ones((1,1)), 'I', shared=False,  sparse=False)
        self.assert_ (self.equal(G1.I[0,0],R))


    def test_weighted_sum_shared (self):
        ''' Check link weighted sum computation with shared kernel '''

        G1 = dana.zeros((5,))
        G2 = dana.ones((5,))
        G1.connect(G2.V,np.ones((5,)), 'I', sparse=False, shared=True)
        G1.dV = 'I'
        G1.compute()
        self.assert_ (self.equal (G1.V, np.array([3,4,5,4,3])))


    def test_weighted_sum_dense (self):
        ''' Check link weighted sum computation with dense kernel '''

        G1 = dana.zeros((5,))
        G2 = dana.ones((5,))
        G1.connect(G2.V,np.ones((5,)), 'I', sparse=False, shared=False)
        G1.dV = 'I'
        G1.compute()
        self.assert_ (self.equal (G1.V, np.array([3,4,5,4,3])))


    def test_weighted_sum_sparse (self):
        ''' Check link weighted sum computation with sparse kernel '''

        G1 = dana.zeros((5,))
        G2 = dana.ones((5,))
        G1.connect(G2.V,np.ones((5,)), 'I', sparse=True)
        G1.dV = 'I'
        G1.compute()
        self.assert_ (self.almost_equal (G1.V, np.array([3,4,5,4,3])))


    def test_distance_shared (self):
        ''' Check link distance computation with shared kernel '''

        G1 = dana.zeros((5,))
        G2 = dana.group(np.random.random((5,)))
        G1.connect(G2.V,np.ones((1,)), 'I-', sparse=False, shared=True)
        G1.dV = 'I'
        self.assertRaises(ValueError, G1.compute)


    def test_distance_dense (self):
        ''' Check link distance computation with dense kernel '''

        G1 = dana.zeros((5,))
        G2 = dana.group(np.random.random((5,)))
        G1.connect(G2.V,np.ones((1,)), 'I-', sparse=False, shared=False)
        G1.dV = 'I'
        G1.compute()
        self.assert_ (self.almost_equal (G1.V, np.abs(1-G2['V'])))


    def test_distance_sparse (self):
        ''' Check link distance computation with sparse kernel '''

        G1 = dana.zeros((5,))
        G2 = dana.group(np.random.random((5,)))
        G1.connect(G2.V,np.ones((1,)), 'I-', sparse=True)
        G1.dV = 'I'
        G1.compute()
        self.assert_ (self.almost_equal (G1.V, np.abs(1-G2['V'])))

    def test_distance_mask (self):
        ''' Check link distance computation with mask'''

        G1 = dana.ones((5,))
        G2 = dana.zeros((5,))
        G1.connect(G2.V,np.ones((1,)), 'I-')
        G1.dV = 'I'
        G1.compute()
        self.assert_ (self.almost_equal (G1.V, np.array([1,1,1,1,1])))


# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(link)
if __name__ == "__main__":
    unittest.main()
