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
"""
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
"""
import numpy as np
import numpy.random as rnd
import scipy as sp

from functions import extract, gaussian
from functions import convolve1d, convolve2d
from functions import zeros, ones, empty
from functions import zeros_like, ones_like, empty_like
from csr_array import csr_array, dot

from clock import Clock, before, after, second, millisecond

from group import Group
from network import Network, run, end, setup, clock

from connection        import Connection, ConnectionError
from dense_connection  import DenseConnection
from sparse_connection import SparseConnection
from shared_connection import SharedConnection

from model         import Model, ModelError
from definition    import Definition, DefinitionError
from equation      import Equation, EquationError
from declaration   import Declaration, DeclarationError
from diff_equation import DifferentialEquation, DifferentialEquationError

from tests import test

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    cdict = {
      'red':  ([0.00, 0, 0], [0.25, 0, 0], [0.50, 1, 1], [0.75, 1, 1], [1.00, 1, 1]),
      'green':([0.00, 0, 0], [0.25, 1, 1], [0.50, 1, 1], [0.75, 1, 1], [1.00, 0, 0]),
      'blue': ([0.00, 1, 1], [0.25, 1, 1], [0.50, 1, 1], [0.75, 0, 0], [1.00, 0, 0]) }
    ice_and_fire = mpl.colors.LinearSegmentedColormap('ice-and-fire',cdict)

    cdict = {
      'red':  ([0.00, 0, 0], [0.50, 0, 0], [1.00, 1, 1]),
      'green':([0.00, 0, 0], [0.50, 1, 1], [1.00, 1, 1]),
      'blue': ([0.00, 1, 1], [0.50, 1, 1], [1.00, 1, 1]) }
    ice = mpl.colors.LinearSegmentedColormap('ice',cdict)

    cdict = {
      'red':  ([0.00, 1, 1], [0.50, 1, 1], [1.00, 1, 1]),
      'green':([0.00, 1, 1], [0.50, 1, 1], [1.00, 0, 0]),
      'blue': ([0.00, 1, 1], [0.50, 0, 0], [1.00, 0, 0]) }
    fire = mpl.colors.LinearSegmentedColormap('fire',cdict)

    cdict = {
      'red':  ([0.00, 0, 0], [1.00, 1, 1]),
      'green':([0.00, 0, 0], [1.00, 0, 0]),
      'blue': ([0.00, 0, 0], [1.00, 0, 0]) }
    dark_red = mpl.colors.LinearSegmentedColormap('dark-red',cdict)

    cdict = {
      'red':  ([0.00, 1, 1], [1.00, 1, 1]),
      'green':([0.00, 1, 1], [1.00, 0, 0]),
      'blue': ([0.00, 1, 1], [1.00, 0, 0]) }
    light_red = mpl.colors.LinearSegmentedColormap('light-red',cdict)

    cdict = {
      'red':  ([0.00, 0, 0], [1.00, 0, 0]),
      'green':([0.00, 0, 0], [1.00, 1, 1]),
      'blue': ([0.00, 0, 0], [1.00, 0, 0]) }
    dark_green = mpl.colors.LinearSegmentedColormap('dark-green',cdict)

    cdict = {
      'red':  ([0.00, 1, 1], [1.00, 0, 0]),
      'green':([0.00, 1, 1], [1.00, 1, 1]),
      'blue': ([0.00, 1, 1], [1.00, 0, 0]) }
    light_green = mpl.colors.LinearSegmentedColormap('light-green',cdict)

    cdict = {
      'red':  ([0.00, 0, 0], [1.00, 0, 0]),
      'green':([0.00, 0, 0], [1.00, 0, 0]),
      'blue': ([0.00, 0, 0], [1.00, 1, 1]) }
    dark_blue = mpl.colors.LinearSegmentedColormap('dark-blue',cdict)

    cdict = {
      'red':  ([0.00, 1, 1], [1.00, 0, 0]),
      'green':([0.00, 1, 1], [1.00, 0, 0]),
      'blue': ([0.00, 1, 1], [1.00, 1, 1]) }
    light_blue = mpl.colors.LinearSegmentedColormap('light-blue',cdict)

except:
    pass

try:
    from info import version as __version__
    from info import release as __release__
except:
    __version__ = 'nobuilt'
    __release__ = None
