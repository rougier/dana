#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Implementation of Gaussian blur on the lena image.


References:
-----------

  http://en.wikipedia.org/wiki/Gaussian_blur

'''
import Image
from dana import *

image = np.asarray(Image.open('lena.jpg'))/256.0
I = image.copy().view(dtype=[('R',float), ('G',float), ('B',float)]).squeeze()
G = gaussian((10,10),.5)
K = G/G.sum()
SharedConnection(I['R'], I['R'], K).propagate()
SharedConnection(I['G'], I['G'], K).propagate()
SharedConnection(I['B'], I['B'], K).propagate()

Z = I.view(dtype=float).reshape(I.shape[0],I.shape[1],3)
Image.fromarray((Z*256).astype(np.uint8)).save('lena-blur.png')

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title('Original image')
plt.imshow(image, origin='upper', interpolation='bicubic')
plt.subplot(1,2,2), plt.title('Gaussian blur')
plt.imshow(I.view(float).reshape(I.shape+(3,)),
           origin='upper', interpolation='bicubic')
plt.show()
