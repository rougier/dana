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


G = zeros(100,'''r; theta;
                 x = r*sin(theta)
                 y = r*cos(theta)''')
G.r = np.random.random(100)
G.theta = 2*np.pi*np.random.random(100)
G.run(n=1)

x,y,r,theta = G.x[0], G.y[0], G.r[0], G.theta[0]
print r, theta
print x, y
print r*np.sin(theta), r*np.cos(theta)
