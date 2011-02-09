#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009,2010,2011 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
from dana import *

G = zeros((3,3), [('x',float), ('y',float)])
A = G.asarray()

A = np.zeros((3,3), [('x',float), ('y',float)])
G = Group(A)
