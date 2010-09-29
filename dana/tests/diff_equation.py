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
from dana import DifferentialEquation, DifferentialEquationError

class TestDifferentialEquationParsing(unittest.TestCase):
    def test_standard_form_1(self):
        eq = DifferentialEquation('dy/dt = 1.0')
        assert eq._varname == 'y'
        assert eq._dtype   == 'float'
    def test_standard_form_2(self):
        eq = DifferentialEquation('dy/dt = 1.0 : float')
        assert eq._varname == 'y'
        assert eq._dtype  == 'float'
    def test_standard_form_3(self):
        eq = DifferentialEquation('dy/dt = 1.0 : int')
        assert eq._varname == 'y'
        assert eq._dtype  == 'int'
    def test_standard_form_4(self):
        eq = DifferentialEquation('dy/dt = 1.2+(3.4)*y')
        assert eq._varname == 'y'
        assert eq._dtype == 'float'
    def test_exponential_form_1(self):
        eq = DifferentialEquation('dy/dt = 1.2+(3.4)*y')
        assert eq._A_string == '1.2'
        assert eq._B_string == '3.4'
    def test_exponential_form_2(self):
        eq = DifferentialEquation('dy/dt = +(1.1)*y')
        assert eq._A_string == '0'
        assert eq._B_string == '1.1'
    def test_exponential_form_3(self):
        eq = DifferentialEquation('dy/dt = (1.3)*y')
        assert eq._A_string == '0'
        assert eq._B_string == '1.3'
    def test_empty_definition(self):
        def test(): eq = DifferentialEquation('')
        self.assertRaises(DifferentialEquationError,test)
    def test_definition_not_valid(self):
        def test(): eq = DifferentialEquation('y=1')
        self.assertRaises(DifferentialEquationError,test)


class TestIntegration(unittest.TestCase):

    def test_diff_equation_integration_euler(self):
        eq = DifferentialEquation('dy/dt = y : float')
        eq.select("Forward Euler")
        y, t,dt = 1.0, 1.0,  0.00001
        for i in range(int(t/dt)):
            y = eq.evaluate(y, dt)
        assert abs(y-np.exp(1)) < 0.0001

    def test_diff_equation_exponential_euler(self):
        eq = DifferentialEquation('dy/dt = 0+(1)*y : float')
        eq.select("Exponential Euler")
        y, t,dt = 1.0, 1.0,  0.01
        for i in range(int(t/dt)):
            y = eq.evaluate(y, dt)
        assert abs(y-np.exp(1)) < 0.0001

    def test_diff_equation_runge_kutta_2(self):
        eq = DifferentialEquation('dy/dt = y : float')
        eq.select("Runge Kutta 2")
        y, t,dt = 1.0, 1.0,  0.001
        for i in range(int(t/dt)):
            y = eq.evaluate(y, dt)
        assert abs(y-np.exp(1)) < 0.0001

    def test_diff_equation_runge_kutta_4(self):
        eq = DifferentialEquation('dy/dt = y : float')
        eq.select("Runge Kutta 4")
        y, t,dt = 1.0, 1.0,  0.01
        for i in range(int(t/dt)):
            y = eq.evaluate(y, dt)
        assert abs(y-np.exp(1)) < 0.0001



if __name__ == "__main__":
    unittest.main()
