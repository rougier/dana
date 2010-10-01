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

G = zeros((5,5), 'V=1')
G.run(n=1)
print G.V

G = zeros((5,5), 'V=1')
G.run(n=1, asynchrony_level=0.1) # 10% of units won't be updated
print G.V
