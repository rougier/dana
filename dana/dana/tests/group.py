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
        self.assertEqual (np.all(G['V'] == np.zeros((2,2))), True)

    def test_dtype_setting (self):
        ''' Check group dtype setting '''
        G = dana.group((2,2), dtype=int)
        self.assertEqual (G['V'].dtype, int)

    def test_parent (self):
        ''' Check group field parent '''
        G = dana.group((2,2), dtype=int)
        self.assertEqual (G.V.parent, G)

    def test_dtype_multisetting (self):
        ''' Check group dtype multisetting '''
        G = dana.group((2,2), dtype=[('V',float), ('U',int)])
        self.assertEqual (G['V'].dtype, float)
        self.assertEqual (G['U'].dtype, int)

    def test_shape (self):
        ''' Check group shape '''
        G = dana.group((2,2))
        self.assertEqual (G.shape, (2,2))
        self.assertEqual (G['V'].shape, (2,2))

    def test_zeros (self):
        ''' Check group basic creation routine 'zeros' '''
        G = dana.zeros((2,2))
        self.assertEqual (np.all(G['V'] == np.zeros((2,2))), True)

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
        self.assertEqual (np.all(G['V'] == np.zeros((2,2))), True)

    def test_ones_like (self):
        ''' Check group basic creation routine 'ones_like' '''
        G = dana.ones_like(np.zeros((2,2)))
        self.assertEqual (np.all(G['V'] == np.ones((2,2))), True)

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



# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(group)
if __name__ == "__main__":
    unittest.main()
