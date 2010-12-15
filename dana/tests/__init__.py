#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import dana
import unittest
from group import *
from model import *
from equation import *
from learning import *
from csr_array import *
from declaration import *
from diff_equation import *
from dense_connection import *
from sparse_connection import *
from shared_connection import *


def test():
    suite = unittest.TestLoader().loadTestsFromModule(dana.tests)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    test()
