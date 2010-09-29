#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
DANA is a python computing framework based on numpy and scipy libraries.

The computational  paradigm supporting  the DANA framework  is grounded  on the
notion of  a unit that is  a set of arbitrary  values that can  vary along time
under the influence of other units and learning. Each unit can be linked to any
other unit (including itself) using a weighted link and a group is a structured
set of such homogeneous units. The  dana framework offers a set of core objects
needed to design  and run such models. However, what is  actually computed by a
unit and what is learned is the  responsibility of the modeler who is in charge
of describing  the equation  governing the behavior  of units groups  over time
and/or learning.
'''
import numpy as np
import numpy.random as rnd

from core import *
from core.group import zeros, ones, empty
from group import Group
from network import Network, run
from connection import Connection, ConnectionError
from dense_connection import DenseConnection
from sparse_connection import SparseConnection
from shared_connection import SharedConnection
from tests import test

try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    pass
try:
    from info import version as __version__
    from info import release as __release__
except:
    __version__ = 'nobuilt'
    __release__ = None
