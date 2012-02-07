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
import Image
from dana import *
from numpy import exp
from display import *


def gabor(shape=(256,256), wavelength=5, angle=0, offset=0, aspect=0.5, bandwidth=1):
    sigma = wavelength/np.pi*np.sqrt(np.log(2)/2)*(2**bandwidth+1)/(2**bandwidth-1)
    sigma_x = sigma
    sigma_y = sigma/float(aspect)
    X,Y = np.meshgrid(np.linspace(-shape[0]/2, +shape[0]/2, shape[0]),
                      np.linspace(-shape[1]/2, +shape[1]/2, shape[1]))
    x,y = X*np.cos(angle)+Y*np.sin(angle), -X*np.sin(angle)+Y*np.cos(angle)
    return np.exp(-0.5*(x**2/sigma_x**2+y**2/sigma_y**2))*np.cos(2*np.pi/wavelength*x+offset)

filename = 'bars-angle.png'
#filename = 'lena.jpg'
image = np.asarray(Image.open(filename).convert('RGB').resize((128,128)))/256.0
image = image.view(dtype=[('r',float), ('g',float), ('b',float)]).squeeze()
image = image[::-1,::]

r, g, b = image['r'], image['g'], image['b']
I = (0.212671*r + 0.715160*g + 0.072169*b)
R,G,B,Y = r - (g+b)/2, g-(r+b)/2, b-(r+g)/2, (r+g)/2-np.abs(r-g)/2-b

V1 = Group(I.shape, '''RG_on  = 1/(1+exp(-(+R_center - G_surround)))
                       GR_on  = 1/(1+exp(-(+G_center - R_surround)))
                       RG_off = 1/(1+exp(-(-R_center + G_surround)))
                       GR_off = 1/(1+exp(-(-G_center + R_surround)))

                       BY_on  = 1/(1+exp(-(+B_center - Y_surround)))
                       YB_on  = 1/(1+exp(-(+Y_center - B_surround)))
                       BY_off = 1/(1+exp(-(-B_center + Y_surround)))
                       YB_off = 1/(1+exp(-(-Y_center + B_surround)))

                       R_center; R_surround;
                       G_center; G_surround;
                       B_center; B_surround;
                       Y_center; Y_surround;''')

shape = (64,64)
scale = 0.05
size  = 0.05
SharedConnection(R, V1('R_center'),  scale*gaussian(shape, 1*size))
SharedConnection(R, V1('R_surround'),scale*gaussian(shape, 3*size))
SharedConnection(G, V1('G_center'),  scale*gaussian(shape, 1*size))
SharedConnection(G, V1('G_surround'),scale*gaussian(shape, 3*size))
SharedConnection(B, V1('B_center'),  scale*gaussian(shape, 1*size))
SharedConnection(B, V1('B_surround'),scale*gaussian(shape, 3*size))
SharedConnection(Y, V1('Y_center'),  scale*gaussian(shape, 1*size))
SharedConnection(Y, V1('Y_surround'),scale*gaussian(shape, 3*size))

V2 = Group(I.shape, '''V_0   = 1/(1+exp(-O_0))
                       V_45  = 1/(1+exp(-O_45))
                       V_90  = 1/(1+exp(-O_90))
                       V_135 = 1/(1+exp(-O_135))

O_0; O_45; O_90; O_135;''')
for d in range(4):
    key = 'O_%d' % (d*45)
    SharedConnection(I, V2(key), gabor((64,64), 5, d*np.pi/4.,  0, .5))
        
run(n=1)

mpl.rcParams['axes.titlesize']      = 'small'
mpl.rcParams['image.cmap']          = 'gray'
mpl.rcParams['image.origin']        = 'upper'
mpl.rcParams['image.interpolation'] = 'nearest'

fig = plt.figure(figsize=(10,10), facecolor='white')

plot(plt.subplot(4, 4,  1), I, 'Intensity')
plot(plt.subplot(4, 4,  2), R, 'Red')
plot(plt.subplot(4, 4,  3), G, 'Blue')
plot(plt.subplot(4, 4,  4), B, 'Green')

plot(plt.subplot(4, 4, 5), V2('O_0'),   u'0°')
plot(plt.subplot(4, 4, 6), V2('O_45'),  u'45°')
plot(plt.subplot(4, 4, 7), V2('O_90'),  u'90°')
plot(plt.subplot(4, 4, 8), V2('O_135'), u'135°')

plot(plt.subplot(4, 4,  9), V1('RG_on'),  u'RG-on')
plot(plt.subplot(4, 4, 10), V1('GR_on'),  u'GR-on')
plot(plt.subplot(4, 4, 11), V1('RG_off'), u'RG-off')
plot(plt.subplot(4, 4, 12), V1('GR_off'), u'GR-off')

plot(plt.subplot(4, 4, 13), V1('BY_on'),  u'BY-on')
plot(plt.subplot(4, 4, 14), V1('YB_on'),  u'YB-on')
plot(plt.subplot(4, 4, 15), V1('BY_off'), u'BY-off')
plot(plt.subplot(4, 4, 16), V1('YB_off'), u'YB-off')

plt.connect('button_press_event', button_press_event)
plt.show()

