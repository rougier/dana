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
from dana import Model, ModelError
from dana import DifferentialEquation, Equation, Declaration

class TestModelDefault(unittest.TestCase):
    def test_1(self):
        U = Model('V')
        assert type(U.V) is Declaration
    def test_2(self):
        U = Model('V=1')
        assert type(U.V) is Equation
    def test_3(self):
        U = Model('dV/dt=1')
        assert type(U.V) is DifferentialEquation
    def test_4(self):
        U = Model('V')
        assert U.V.dtype == 'float'
    def test_5(self):
        U = Model('V=1')
        assert U.V.dtype == 'float'
    def test_6(self):
        U = Model('dV/dt=1')
        assert U.V.dtype == 'float'

class TestModelError(unittest.TestCase):
    def test_1(self):
        def test(): U = Model('V=1; V=1')
        self.assertRaises(ModelError,test)
    def test_2(self):
        def test(): U = Model('V;V')
        self.assertRaises(ModelError,test)
    def test_3(self):
        def test(): U = Model('dV/dt=1;dV/dt=1')
        self.assertRaises(ModelError,test)
    def test_4(self):
        def test(): U = Model('V=1; V')
        self.assertRaises(ModelError,test)
    def test_5(self):
        def test(): U = Model('dV/dt=1; V')
        self.assertRaises(ModelError,test)
    def test_6(self):
        def test(): U = Model('dV/dt=1; V=1')
        self.assertRaises(ModelError,test)

class TestModelEquationOrdering(unittest.TestCase):
    def test_1(self):
        def test(): U = Model('A=B; B=C; C=A')
        self.assertRaises(ModelError,test)
    def test_2(self):
        U = Model('B=A; A=C; C=1')
        v = [eq.varname for eq in U._equations]
        assert v == ['C','A','B']

class TestEvaluation(unittest.TestCase):
    def test_1(self):
        model = Model('''dx/dt = 1.0
                          y = x''')
        namespace = {'x' : 0, 'y' : 0}
        t, dt = 1.0, 0.001
        for i in range(int(t/dt)):
            model.run(namespace, dt=dt)
        assert (namespace['x']-1.0) < 1e-15
        assert (namespace['y']-1.0) < 1e-15

if __name__ == "__main__":
    unittest.main()
