#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Implementation of the Sobel operator on the lena image.


References:
-----------

  http://en.wikipedia.org/wiki/Sobel_operator

'''
import Image
from dana import *

image = np.asarray(Image.open('lena.jpg'))/256.0
I = image.view(dtype=[('R',float), ('G',float), ('B',float)]).squeeze()
L = (0.212671*I['R'] + 0.715160*I['G'] + 0.072169*I['B'])
src = Group(I.shape, '''V = sqrt(Gx**2+Gy**2) : float
                        Gx                    : float
                        Gy                    : float ''')
Kx = np.array([[-1., 0.,+1.], [-2., 0.,+2.], [-1., 0., 1.]])
Gx = SharedConnection(L, src('Gx'), Kx)
Ky = np.array([[+1.,+2.,+1.], [ 0., 0., 0.], [-1.,-2.,-1.]])
Gy = SharedConnection(L, src('Gy'), Ky)  
src.run(n=1)

Z = I.view(dtype=float).reshape(I.shape[0],I.shape[1],3)
Image.fromarray((src.V*256).astype(np.uint8)).save('lena-sobel.png')

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title('Original image')
plt.imshow(image, origin='upper', interpolation='bicubic')
plt.subplot(1,2,2), plt.title('Sobel filter')
plt.imshow(src.V, origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
plt.show()
