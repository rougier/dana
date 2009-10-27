#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# glumpy - Fast OpenGL numpy visualization
# Copyright (c) 2009 - Nicolas P. Rougier
#
# This file is part of glumpy.
#
# glumpy is free  software: you can redistribute it and/or  modify it under the
# terms of  the GNU General  Public License as  published by the  Free Software
# Foundation, either  version 3 of the  License, or (at your  option) any later
# version.
#
# glumpy is  distributed in the  hope that it  will be useful, but  WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy  of the GNU General Public License along with
# glumpy. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------
import dana
import numpy as np


def cartesian(rho, theta):
    ''' Polar to cartesian coordinates. '''
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    return x,y

def logpolar(rho, theta):
    ''' Polar to logpolar coordinates. '''
    A = 3.0  # Shift in the SC mapping function in deg 
    Bx = 1.4 # Collicular magnification along u axe in mm/rad 
    By = 1.8 # Collicular magnification along v axe in mm/rad 
    xmin, xmax = 0.0, 4.80743279742
    ymin, ymax = -2.76745559565, 2.76745559565
    rho = rho*90.0
    x = Bx*np.log(np.sqrt(rho*rho+2*A*rho*np.cos(theta)+A*A)/A)
    y = By*np.arctan(rho*np.sin(theta)/(rho*np.cos(theta)+A))
    x = (x-xmin)/(xmax-xmin)
    y = (y-ymin)/(ymax-ymin)
    return x,y


def retinotopy(Rs,Ps):
    ''' '''
    s = 4
    rho = ((np.logspace(start=0, stop=1, num=s*Rs[1],base=10)-1)/9.)
    theta = np.linspace(start=-np.pi/2,stop=np.pi/2, num=s*Rs[0])
    rho = rho.reshape((s*Rs[1],1))
    rho = np.repeat(rho,s*Rs[0], axis=1)
    theta = theta.reshape((1,s*Rs[0]))
    theta = np.repeat(theta,s*Rs[1], axis=0)
    y,x = cartesian(rho,theta)
    a,b = x.min(), x.max()
    x = (x-a)/(b-a)
    a,b = y.min(), y.max()
    y = (y-a)/(b-a)

    Px = np.ones(Ps, dtype=int)*0
    Py = np.ones(Ps, dtype=int)*0

    xi = (x*(Rs[0]-1)).astype(int)
    yi = ((0.5+0.5*y)*(Rs[1]-1)).astype(int)
    yc,xc = logpolar(rho,theta)
    a,b = xc.min(), xc.max()
    xc = (xc-a)/(b-a)
    a,b = yc.min(), yc.max()
    yc = (yc-a)/(b-a)
    xc = (xc*(Ps[0]-1)).astype(int)
    yc = ((.5+yc*0.5)*(Ps[1]-1)).astype(int)
    Px[xc,yc] = xi
    Py[xc,yc] = yi

    xi = (x*(Rs[0]-1)).astype(int)
    yi = ((0.5-0.5*y)*(Rs[1]-1)).astype(int)
    yc,xc = logpolar(rho,theta)
    a,b = xc.min(), xc.max()
    xc = (xc-a)/(b-a)
    a,b = yc.min(), yc.max()
    yc = (yc-a)/(b-a)
    xc = (xc*(Ps[0]-1)).astype(int)
    yc = (((1-yc)*0.5)*(Ps[1]-1)).astype(int)
    Px[xc,yc] = xi
    Py[xc,yc] = yi

    return Px, Py


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    from PIL import Image
    import pylab
    import dana.pylab
    
    image = Image.open('lena-grey.png')
    Scene  = np.asarray(image, dtype=np.float32)/256. # Visual scene
    Retina = np.zeros((128,128),dtype=np.float32)     # Retina
    V1     = np.zeros((128,128),dtype=np.float32)     # "V1"
    Px,Py  = retinotopy(Retina.shape,V1.shape)

    Retina[...] = Scene[64:64+Retina.shape[0], 64:64+Retina.shape[1]]
    V1[...] = Retina[Px,Py]
    view = dana.pylab.view([Scene, Retina, V1], vmin=0,
                           origin='upper', cmap=pylab.cm.gray)
    view.show()
    

