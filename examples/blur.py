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
I = image.view(dtype=[('R',float), ('G',float), ('B',float)]).squeeze()

src = Group(I.shape, 'R; G; B')
G = gaussian((10,10),.5)
K = G/G.sum()
Cr = SharedConnection(I['R'], src('R'), K)
Cg = SharedConnection(I['G'], src('G'), K)
Cb = SharedConnection(I['B'], src('B'), K)
src.run(1)

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title('Original image')
plt.imshow(image, origin='upper', interpolation='bicubic')
plt.subplot(1,2,2), plt.title('Gaussian blur')
plt.imshow(src.asarray().view(float).reshape(src.shape+(3,)),
           origin='upper', interpolation='bicubic')
plt.show()
