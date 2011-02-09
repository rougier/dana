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

x0, v0, a = 1.0, 2.0, 3.0
t, dt = 1.0, 0.01
G = zeros(1,'dx/dt=v; dv/dt=a')
G.x, G.v = x0, v0
t, dt = 1.0, 0.01
run(t=t, dt=dt)
print G.x[0]
print 0.5*a*t**2 + v0*t + x0
