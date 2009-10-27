#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#   ____  _____ _____ _____ 
#  |    \|  _  |   | |  _  |   DANA, Distributed Asynchronous Adaptive Numerical
#  |  |  |     | | | |     |         Computing Framework
#  |____/|__|__|_|___|__|__|         Copyright (C) 2009 INRIA  -  CORTEX Project
#                         
#  This program is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free Software
#  Foundation, either  version 3 of the  License, or (at your  option) any later
#  version.
# 
#  This program is  distributed in the hope that it will  be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public  License for  more
#  details.
# 
#  You should have received a copy  of the GNU General Public License along with
#  this program. If not, see <http://www.gnu.org/licenses/>.
# 
#  Contact: 
# 
#      CORTEX Project - INRIA
#      INRIA Lorraine, 
#      Campus Scientifique, BP 239
#      54506 VANDOEUVRE-LES-NANCY CEDEX 
#      FRANCE
# 
import numpy as np
import dana

n,m  = 100,100
y,x  = 0.321, 0.123

Z = dana.gaussian((n,m), .25, (y,x))

Zx = np.linspace(-1.0, 1.0, m)
Zx = np.resize(Zx,(n,m))
Zy = np.linspace(-1.0, 1.0, n).transpose()
Zy = np.resize(Zy,(m,n)).transpose()

print 'x = %.5f, decoded x = %.5f' %(x, (Z*Zx).sum()/(Z.sum()))
print 'y = %.5f, decoded y = %.5f' %(y, (Z*Zy).sum()/(Z.sum()))
