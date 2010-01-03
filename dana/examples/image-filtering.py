#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
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
''' This example shows how to implement basic image filter using DANA.
'''
import dana
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Load image
I = np.asarray(Image.open('lena.png'))/256.0
image = dana.group(I.shape[:-1], keys=['R','G','B'])
image.R = I[:,:,0]
image.G = I[:,:,1]
image.B = I[:,:,2]

# Create groups for luminance conversion (G), sobel filter (S)
#  and gaussian blur(B)
G = dana.zeros(I.shape[:-1])
S = dana.zeros(I.shape[:-1])
B = dana.zeros(I.shape[:-1], keys=['R','G','B'])

# Connect luminance group to R,G,B channels
G.connect(image.R, np.ones((1,1)), 'R', shared=True)
G.connect(image.G, np.ones((1,1)), 'G', shared=True)
G.connect(image.B, np.ones((1,1)), 'B', shared=True)

# Connect filter group to luminance
S.connect(G.V, np.array([[-1, 0,+1],
                         [-2, 0,+2],
                         [-1, 0, 1]]), 'Gx', shared=True)
S.connect(G.V, np.array([[+1,+2,+1],
                         [ 0, 0, 0],
                         [-1,-2,-1]]), 'Gy', shared=True)

# Connect filter group to blur
K = np.array([[1, 4, 7, 4,1],
              [4,16,26,16,4],
              [7,26,41,16,7],
              [4,16,26,16,4],
              [1, 4, 7, 4,1]])/273.
B.connect(image.R, K, 'Gr', shared=True)
B.connect(image.G, K, 'Gg', shared=True)
B.connect(image.B, K, 'Gb', shared=True)

# Compute luminance
G.dV = '-V + 0.212671*R + 0.715160*G + 0.072169*B'
G.compute()

# Compute Sobel filters
S.dV = '-V + sqrt(Gx*Gx+Gy*Gy)'
S.compute()

# Compute Gaussian blur
B.dR = 'Gr'
B.dG = 'Gg'
B.dB = 'Gb'
B.compute()

# Show results
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Original image')
plt.xticks([])
plt.yticks([])
plt.imshow(I, origin='upper', interpolation='bicubic')
plt.subplot(1,3,2)
plt.title('Sobel filter')
plt.imshow(S.V, origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,3)
plt.title('Gaussian blur')
plt.imshow(B.asarray().view(float).reshape(B.shape+(3,)),
           origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()
