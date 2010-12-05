#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Show how to evalute a model asynchronously

'''
from dana import *

G = zeros((5,5), 'dV/dt=1')
G.run(t=1.0)
print G.V
print

G = zeros((5,5), 'dV/dt=1')
# 80% of units are evaluated at each timestep
G.run(t=1.0, asynchrony_level=0.2)
print G.V
