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
# Implementation of :
#   Robinson, D. A.
#   Eye movements evoked by collicular stimulation in the alert monkey.
#   Vision Res 12 (11), 1795-1808, November 1972.
from pylab import *


A = 3.0 # Shift in the SC mapping function in deg 
Bx = 1.4 # Collicular magnification along u axe in mm/rad 
By = 1.8 # Collicular magnification along v axe in mm/rad 
xmin, xmax = 0.0, 4.80743279742
ymin, ymax = -2.76745559565, 2.76745559565


R = [1, 2, 3, 5, 10, 45, 90]
n = 9
figure(figsize=(8,8))

X,Y = zeros((n,)), zeros((n,))
for j in range(len(R)):
    for i in range(n):
        theta = ((i/float(n-1))*math.pi-math.pi/2.0)
        r = R[j]
        X[i] = Bx*log(sqrt(r*r+2*A*r*cos(theta)+A*A)/A)
        Y[i] = By*arctan(r*sin(theta)/(r*cos(theta)+A))
    plot(X, Y, linewidth=1, color='b')
    text(X[n/2], 0, u'%s°' % repr(r),
         color='b',
         rotation=-90, fontsize = 14,
         horizontalalignment='center',
         verticalalignment='center',
         bbox=dict(ec='b', fc= 'w', alpha=1.0))

m = 200
X,Y = zeros((m,)), zeros((m,))
for i in range(n):
    for j in range(m):
        theta = ((i/float(n-1))*math.pi-math.pi/2.0)
        r = j/float(m)*90.0
        X[j] = Bx*log(sqrt(r*r+2*A*r*cos(theta)+A*A)/A)
        Y[j] = By*arctan(r*sin(theta)/(r*cos(theta)+A))
    plot(X,Y,linewidth=1, color='b')

title('Cortical Magnification', fontsize=16)
axis([xmin, xmax, ymin, ymax])
grid()
show()
