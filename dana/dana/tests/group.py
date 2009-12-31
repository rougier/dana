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

class group (unittest.TestCase):
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


    def test_default_key (self):
        ''' Check group default key is 'V' '''
        G = dana.group((2,2))
        self.assertEqual ('V' in G.dtype.names, True)

    def test_key_naming (self):
        ''' Check group key naming '''
        G = dana.group((2,2), keys = ['U', 'W'])
        self.assertEqual ('U' in G.dtype.names, True)
        self.assertEqual ('W' in G.dtype.names, True)

    def test_key_access (self):
        ''' Check group key access '''
        G = dana.zeros((2,2))
        self.assertEqual (np.all(G.V == np.zeros((2,2))), True)

    def test_dtype_setting (self):
        ''' Check group dtype setting '''
        G = dana.group((2,2), dtype=int)
        self.assertEqual (G.V.dtype, int)

    def test_parent (self):
        ''' Check group field parent '''
        G = dana.group((2,2), dtype=int)
        self.assertEqual (G.V.parent, G)

    def test_dtype_multisetting (self):
        ''' Check group dtype multisetting '''
        G = dana.group((2,2), dtype=[('V',float), ('U',int)])
        self.assertEqual (G.V.dtype, float)
        self.assertEqual (G.U.dtype, int)

    def test_shape (self):
        ''' Check group shape '''
        G = dana.group((2,2))
        self.assertEqual (G.shape, (2,2))
        self.assertEqual (G.V.shape, (2,2))

    def test_reshape (self):
        ''' Check group reshape '''
        G = dana.zeros((2,2))
        G1 = G.reshape((4,1))
        G2 = G.reshape((1,4))
        self.assertEqual (G1.shape,   (4,1))
        self.assertEqual (G1.V.shape, (4,1))
        self.assertEqual (G2.shape,   (1,4))
        self.assertEqual (G2.V.shape, (1,4))
        self.assertEqual (id(G.V), id(G1.V.base))
        self.assertEqual (id(G.V), id(G2.V.base))

    def test_zeros (self):
        ''' Check group basic creation routine 'zeros' '''
        G = dana.zeros((2,2))
        self.assertEqual (np.all(G.V == np.zeros((2,2))), True)

    def test_ones (self):
        ''' Check group basic creation routine 'ones' '''
        G = dana.ones((2,2))
        self.assertEqual (np.all(G.V == np.ones((2,2))), True)

    def test_empty (self):
        ''' Check group basic creation routine 'empty' '''
        G = dana.empty((2,2))
        self.assertEqual (G.shape, (2,2))

    def test_zeros_like (self):
        ''' Check group basic creation routine 'zeros_like' '''
        G = dana.zeros_like(np.ones((2,2)))
        self.assertEqual (np.all(G.V == np.zeros((2,2))), True)

    def test_ones_like (self):
        ''' Check group basic creation routine 'ones_like' '''
        G = dana.ones_like(np.zeros((2,2)))
        self.assertEqual (np.all(G.V == np.ones((2,2))), True)

    def test_creation (self):
        ''' Check group creation '''
        G = dana.group((2,))
        self.assertEqual (G.shape, (2,))
        G = dana.group((2,2))
        self.assertEqual (G.shape, (2,2))
        G = dana.group((2,2,2))
        self.assertEqual (G.shape, (2,2,2))
        G = dana.group(np.ones((2,2)))
        self.assertEqual (G.shape, (2,2))

    def test_mask (self):
        ''' Check group mask '''
        G = dana.zeros((2,2))
        G.mask[0,0] = False
        V = np.zeros((2,2))
        V[0,0] = np.NaN
        self.assert_ (self.equal(G.V,V))

    def test_set_dead_unit (self):
        ''' Check set dead unit '''
        G = dana.zeros((2,2))
        G.mask[0,0] = False
        G.V[0,0] = 1
        V = np.zeros((2,2))
        V[0,0] = np.NaN
        self.assert_ (self.equal(G.V,V))


# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(group)
if __name__ == "__main__":
    unittest.main()
