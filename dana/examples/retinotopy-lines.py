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


def project(rho=1, theta=[-np.pi/2,np.pi/2], func=cartesian, n=200):
    X,Y = [],[]
    if type(rho) in [int,float]:
        for i in range(n+1):
            angle = theta[0] + i*(theta[1]-theta[0])/float(n)
            x,y = func(rho,angle)
            X.append(x)
            Y.append(y)
    elif type(theta) in [int,float]:
        for i in range(n+1):
            r = rho[0] + i*(rho[1]-rho[0])/float(n)
            x,y = func(r,theta)
            X.append(x)
            Y.append(y)
    return np.array(X),np.array(Y)
    

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt


    # Simulation parameters
    # ______________________________________________________________________________
    r       = 255
    N       = 127
    dt      = 0.1
    alpha   = 10.0
    tau     = 1.0
    h       = 0.0    


    # Build groups
    # ______________________________________________________________________________
    retina = dana.zeros((r,r))
    V1 = dana.zeros((N,N),dtype=np.float32)
    Px,Py = retinotopy(retina.shape,V1.shape)
    colliculus = dana.zeros((N,N), name='focus')


    # Connections
    # ______________________________________________________________________________
    Wi = np.ones((1,1))
    colliculus.connect(V1.V, Wi, 'I', shared=True)
    Wf = 1.25*dana.gaussian((2*N+1,2*N+1), 0.1) - 0.675*dana.gaussian((2*N+1,2*N+1), 1)
    colliculus.connect (colliculus.V, Wf, 'L', shared=True)


    # Set Dynamic Neural Field equation
    # ______________________________________________________________________________
    colliculus.dV = 'minimum(maximum(V+dt/tau*(-V+(L/(N*N)*40*40+I+h)/alpha),0),1)'


    # Set input
    # ______________________________________________________________________________
    retina.V = dana.gaussian((r,r), width=.02, center=(0.0,0.15)) \
               + np.random.random(retina.shape)*.2
    v = retina.V[0,0].copy()
    retina.V[0,0] = 0
    V1.V = retina.V[Px,Py]
    retina.V[0,0] = v


    # Run some iterations
    # ______________________________________________________________________________
    n = 250
    for i in range(n):
        colliculus.compute(dt)

    # Display results
    # ______________________________________________________________________________
    fig = plt.figure(figsize=(21,7),
                     facecolor='w', edgecolor='k',
                     subplotpars=matplotlib.figure.SubplotParams(left=0.03, right=0.97,
                                                                 bottom=0.03, top=0.97,
                                                                 wspace=0.05, hspace=0.05))
    color = (0,0,0)
    alpha = 0.5
    interpolation='bicubic'
    cmap = plt.cm.gray_r
    #cmap = plt.cm.hot


    ax = plt.subplot(131, axisbg = 'w')
    for rho in [5,15,45,90]:
        X,Y = project(rho/90.0, [-np.pi/2, np.pi/2], cartesian)
        plt.plot(X, Y, color=color, alpha=alpha)
        X,Y = project(rho/90.0, [np.pi/2, 3*np.pi/2], cartesian)
        plt.plot(X,Y, color=color, alpha=alpha)
    for theta in [-45,0,45,90]:
        angle = theta/90.0*np.pi/2
        X,Y = project([0,1], angle, cartesian)
        plt.text(X[-1]*1.025, Y[-1]*1.025,u'%d°' % theta,
                 color=color, alpha=alpha,
                 horizontalalignment='left', verticalalignment='bottom')
        plt.plot(X, Y, color=color, alpha=alpha)
        X,Y = project([0,1], angle, cartesian)
        plt.plot(-X,-Y, color=color, alpha=alpha)
        plt.text(-X[-1]*1.025, -Y[-1]*1.025,u'%d°' % -theta,
                 color=color, alpha=alpha,
                 horizontalalignment='right', verticalalignment='top')

    plt.imshow(retina.V,extent=[-1,1,-1,1], interpolation=interpolation, cmap=cmap)
    plt.xlim(-1.15,1.15)
    plt.ylim(-1.15,1.15)
    plt.xticks([])
    plt.yticks([])
    plt.title("Retina")
    ax.set_aspect(1)


    ax = plt.subplot(132, axisbg = 'w')
    for rho in [5,15,45,90]:
        X,Y = project(rho/90.0, [-np.pi/2, np.pi/2], logpolar)
        plt.plot(X, Y*2-1, color=color, alpha=alpha)
        plt.text(X[-1]*1.0, (Y[-1]*2-1)*1.05,u'%d°' % rho,
                 color=color, alpha=alpha,
                 horizontalalignment='right', verticalalignment='bottom')
        X,Y = project(rho/90.0, [-np.pi/2, np.pi/2], logpolar)
        plt.plot(-X, Y*2-1, color=color, alpha=alpha)
        plt.text(-X[-1], (Y[-1]*2-1)*1.05,u'%d°' % rho,
                  color=color, alpha=alpha,
                  horizontalalignment='left', verticalalignment='bottom')
    for theta in [-90,-45,0,45,90]:
        angle = theta/90.0*np.pi/2
        X,Y = project([0,1], angle, logpolar)
        plt.plot(X,Y*2-1, color=color, alpha=alpha)
        X,Y = project([0,1], angle, logpolar)
        plt.text(X[-1]*1.025, (Y[-1]*2-1),u'%d°' % theta,
                 color=color, alpha=alpha,
                 horizontalalignment='left', verticalalignment='bottom')
        plt.plot(-X,Y*2-1, color=color, alpha=alpha)
        plt.text(-X[-1]*1.025, (Y[-1]*2-1),u'%d°' % theta,
                  color=color, alpha=alpha,
                  horizontalalignment='right', verticalalignment='bottom')

    plt.imshow(V1.V,extent=[-1,1,-1,1], interpolation=interpolation, cmap=cmap)
    plt.xlim(-1.15,1.15)
    plt.ylim(-1.15,1.15)
    plt.xticks([])
    plt.yticks([])
    plt.title("Area V1")
    ax.set_aspect(1)


    ax = plt.subplot(133, axisbg = 'w')
    for rho in [5,15,45,90]:
        X,Y = project(rho/90.0, [-np.pi/2, np.pi/2], logpolar)
        plt.plot(X, Y*2-1, color=color, alpha=alpha)
        plt.text(X[-1]*1.0, (Y[-1]*2-1)*1.05,u'%d°' % rho,
                 color=color, alpha=alpha,
                 horizontalalignment='right', verticalalignment='bottom')
        X,Y = project(rho/90.0, [-np.pi/2, np.pi/2], logpolar)
        plt.plot(-X, Y*2-1, color=color, alpha=alpha)
        plt.text(-X[-1], (Y[-1]*2-1)*1.05,u'%d°' % rho,
                  color=color, alpha=alpha,
                  horizontalalignment='left', verticalalignment='bottom')
    for theta in [-90,-45,0,45,90]:
        angle = theta/90.0*np.pi/2
        X,Y = project([0,1], angle, logpolar)
        plt.plot(X,Y*2-1, color=color, alpha=alpha)
        X,Y = project([0,1], angle, logpolar)
        plt.text(X[-1]*1.025, (Y[-1]*2-1),u'%d°' % theta,
                 color=color, alpha=alpha,
                 horizontalalignment='left', verticalalignment='bottom')
        plt.plot(-X,Y*2-1, color=color, alpha=alpha)
        plt.text(-X[-1]*1.025, (Y[-1]*2-1),u'%d°' % theta,
                  color=color, alpha=alpha,
                  horizontalalignment='right', verticalalignment='bottom')


    plt.imshow(colliculus.V,extent=[-1,1,-1,1], interpolation=interpolation, cmap=cmap)
    plt.xlim(-1.15,1.15)
    plt.ylim(-1.15,1.15)
    plt.xticks([])
    plt.yticks([])
    plt.title("Colliculus")
    ax.set_aspect(1)

    plt.show()


