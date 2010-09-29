#!/usr/bin/env python
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
import unittest
import numpy as np
from dana import Equation, EquationError

class TestEquationParsing(unittest.TestCase):
    def test_1(self):
        eq = Equation('V=1')
        assert type(eq) is Equation
    def test_2(self):
        eq = Equation('V=1:')
        assert type(eq) is Equation
    def test_3(self):
        eq = Equation('V=1:float')
        assert type(eq) is Equation
    def test_4(self):
        def test(): eq = Equation('')
        self.assertRaises(EquationError,test)

class TestEquationIntegration(unittest.TestCase):
    def test_1(self):
        eq = Equation('V = 1 : float')
        assert eq() == 1
    def test_2(self):
        eq = Equation('V = x : float')
        assert eq(x=1) == 1
    def test_3(self):
        def f(x): return x
        eq = Equation('V = f(x) : float')
        assert eq(x=0.5) == 0.5
    def test_4(self):
        eq = Equation('V = sin(x)**2 + cos(x)**2 : float')
        assert eq(x=0.5) == 1
    def test_5(self):
        y, t,dt = 1.0, 1.0,  0.00001
        eq = Equation('y = y*dt')
        for i in range(int(t/dt)):
            y += eq.evaluate(y=y, dt=dt)
        assert abs(y-np.exp(1)) < 0.0001



if __name__ == "__main__":
    unittest.main()
