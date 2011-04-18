#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
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
run(n=1)

Z = I.view(dtype=float).reshape(I.shape[0],I.shape[1],3)
Image.fromarray((src.V*256).astype(np.uint8)).save('lena-sobel.png')

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title('Original image')
plt.imshow(image, origin='upper', interpolation='bicubic')
plt.subplot(1,2,2), plt.title('Sobel filter')
plt.imshow(src.V, origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
plt.show()
