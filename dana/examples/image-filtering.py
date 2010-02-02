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
'''
This example shows how to implement basic image processing using DANA.
'''
import Image, dana, numpy
import matplotlib.pyplot as plt

image = numpy.asarray(Image.open('lena.png'))/256.0
I = dana.group(image.shape[:-1], keys=['L','R','G','B'])
I.R = image[..., 0]
I.G = image[..., 1]
I.B = image[..., 2]
I.L = 0.212671*I.R + 0.715160*I.G + 0.072169*I.B

# Create a group for (S)obel filter
S = dana.zeros(I.shape)
# Connect filter group to luminance
S.connect((I,'L'), numpy.array([[-1, 0,+1],
                                [-2, 0,+2],
                                [-1, 0, 1]]), 'Gx', shared=True)
S.connect((I,'L'), numpy.array([[+1,+2,+1],
                                [ 0, 0, 0],
                                [-1,-2,-1]]), 'Gy', shared=True)
# Compute Sobel filters
S.dV = '-V + sqrt(Gx*Gx+Gy*Gy)'
S.compute()


# Create a group for gaussian (B)lur
B = dana.zeros(I.shape, keys=['R','G','B'])

# Connect filter group to blur
K = numpy.array([[1, 4, 7, 4,1],
                 [4,16,26,16,4],
                 [7,26,41,16,7],
                 [4,16,26,16,4],
                 [1, 4, 7, 4,1]])/273.0

B.connect((I,'R'), K, 'r', shared=True)
B.connect((I,'G'), K, 'g', shared=True)
B.connect((I,'B'), K, 'b', shared=True)

# Compute Gaussian blur
B.dR = '-R +r'
B.dG = '-G +g'
B.dB = '-B +b'
B.compute()

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Original image')
plt.imshow(image, origin='upper', interpolation='bicubic')
plt.subplot(1,3,2)
plt.title('Sobel filter')
plt.imshow(S.V, origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
plt.subplot(1,3,3)
plt.title('Gaussian blur')
plt.imshow(B.asarray().view(float).reshape(B.shape+(3,)),
          origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
plt.show()

