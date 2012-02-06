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
        import numpy as np
        eq = Equation('V = np.sin(x)**2 + np.cos(x)**2 : float')
        assert eq(x=0.5) == 1
    def test_5(self):
        y, t,dt = 1.0, 1.0,  0.00001
        eq = Equation('y = y*dt')
        for i in range(int(t/dt)):
            y += eq.evaluate(y=y, dt=dt)
        assert abs(y-np.exp(1)) < 0.0001



if __name__ == "__main__":
    unittest.main()
