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
from tools import np_equal
from dana import Network, Group, Clock
from dana import SparseConnection, DenseConnection, SharedConnection

class TestEvaluationOrder(unittest.TestCase):
    def test_1(self):
        A = Group(1, "V = B['V']")
        A[...] = 1
        B = Group(1, "V = A['V']")
        B[...] = 2
        A.setup()
        B.setup()
        A.evaluate(dt=1, update=False)
        B.evaluate(dt=1, update=False)
        A.update()
        B.update()
        assert A['V'][0] == 2 and B['V'][0] == 1

    def test_2(self):
        net = Network()
        A = Group(1, "V = B['V']")
        B = Group(1, "V = A['V']")
        A[...] = 1
        B[...] = 2
        net.append(A)
        net.append(B)
        net.run(n=1)
        assert A['V'][0] == 2 and B['V'][0] == 1

    def test_3(self):
        net = Network(Clock(0.0, 1.0, 0.001))
        src = Group((1,), 'dV/dt=1')
        src[...] = 1
        dst = Group((1,) , 'I')
        dst[...] = 0
        kernel = np.ones(1)
        C = DenseConnection(src('V'), dst('I'), kernel)
        net.append(src)
        net.append(dst)
        V,I=[],[]
        @net.clock.every(0.1,order=-1)
        def do(t):
            V.append(src['V'][0])
            I.append(dst['I'][0])
        net.run(time=1.0, dt=0.1)
        assert np_equal(np.array(V),
                        [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])
        assert np_equal(np.array(I),
                        [0.0,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])

if __name__ == "__main__":
    unittest.main()
