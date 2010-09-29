#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Core objects and functions
'''
from group import group
#from group import ones, zeros, empty
#from group import ones_like, zeros_like, empty_like
from model import Model, ModelError
from functions import extract, gaussian
from functions import convolve1d, convolve2d
from csr_array import csr_array, dot
from definition import Definition, DefinitionError
from equation import Equation, EquationError
from declaration import Declaration, DeclarationError
from diff_equation import DifferentialEquation, DifferentialEquationError
