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
