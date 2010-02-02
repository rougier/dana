#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009, 2010 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE
import matplotlib.pyplot as plt
import numpy as np
import dana

n = 100
G = dana.group((n,n), dtype=int)
K = np.array([[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
G.connect(G, K, 'N', sparse=True)
G.dV = '-V+np.maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'
G.V = np.random.randint(0,2,G.shape)
for i in range(50):
    G.compute()
plt.imshow(G.V, cmap=plt.cm.gray_r, extent=[0,n,0,n],
           interpolation='nearest', origin='lower')
plt.show()

