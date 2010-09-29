#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
import unittest
import numpy as np
from dana import Declaration, DeclarationError

class TestDeclaration(unittest.TestCase):
    def test_1(self):
        D = Declaration('V')
        assert type(D) is Declaration
    def test_2(self):
        D = Declaration('V:')
        assert type(D) is Declaration
    def test_3(self):
        D = Declaration('V:float')
        assert type(D) is Declaration
    def test_4(self):
        def test(): D = Declaration('')
        self.assertRaises(DeclarationError,test)

class TestDeclarationEvaluation(unittest.TestCase):
    def test_1(self):
        D = Declaration('V:bool')
        assert D() == bool
    def test_2(self):
        D = Declaration('V:int')
        assert D() == int
    def test_3(self):
        D = Declaration('V:float')
        assert D() == float

if __name__ == "__main__":
    unittest.main()
