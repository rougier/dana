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
import numpy, Image, pylab
import dana, dana.pylab

# Load image
# ______________________________________________________________________________
I = numpy.asarray(Image.open('lena.png'))

# Build groups
# ______________________________________________________________________________
image = dana.group(I.shape[:-1], keys=['R','G','B'])
image.R = I[:,:,0]/256.0
image.G = I[:,:,1]/256.0
image.B = I[:,:,2]/256.0
grey = dana.group(I.shape[:-1])
sobel= dana.group(I.shape[:-1])

# Connections
# ______________________________________________________________________________
grey.connect(image.R, numpy.ones((1,1)), 'R', shared=True)
grey.connect(image.G, numpy.ones((1,1)), 'G', shared=True)
grey.connect(image.B, numpy.ones((1,1)), 'B', shared=True)
sobel.connect(grey.V, numpy.array([[-1, 0,+1],
                                   [-2, 0,+2],
                                   [-1, 0, 1]]), 'Gx', shared=True) 
sobel.connect(grey['V'], numpy.array([[+1,+2,+1],
                                      [ 0, 0, 0],
                                      [-1,-2,-1]]), 'Gy', shared=True)


# Set group equations
# ______________________________________________________________________________
grey.dV  = '0.212671*R + 0.715160*G + 0.072169*B'
sobel.dV = 'sqrt(Gx*Gx+Gy*Gy)'

# Run one iteration
# ______________________________________________________________________________
grey.compute()
sobel.compute()

# Display result using pylab
# __________________________________________________________________________
view = dana.pylab.view([sobel.V],
                       origin='upper', vmin=0, vmax=1, cmap=pylab.cm.gray)
view.show()
