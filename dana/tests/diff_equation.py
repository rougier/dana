#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
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
        assert eq._B_string == '-3.4'
    def test_exponential_form_2(self):
        eq = DifferentialEquation('dy/dt = +(1.1)*y')
        assert eq._A_string == '0'
        assert eq._B_string == '-1.1'
    def test_exponential_form_3(self):
        eq = DifferentialEquation('dy/dt = (1.3)*y')
        assert eq._A_string == '0'
        assert eq._B_string == '-1.3'
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
