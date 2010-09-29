# -*- coding: utf-8 -*-
#
# Dymamic Neural Field with delays
# Copyright (C) 2010 Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#
#  Contributors: Nicolas Rougier, Cyril Noël, Axel Hutt
#
''' Numerical integration of dynamic neural fields with delays

This script implements the numerical integration of a dynamic neural fields
with delays:

  ∂V(x,t)             ⌠
τ ------- = -V(x,t) + ⎮  K(|x-y|)S(V(y,t-|x-y|/c))d²y + I(x,t)
   ∂t                 ⌡Ω

where # V(x,t) is the potential of a neural population at position x and time t
      # K(x) is a neighborhood function from [0,√2l] -> ℝ
      # S(u) is the firing rate of a single neuron from  ℝ⁺ -> ℝ
      # c is the velocity of an action potential (m/s)
      # τ is the temporal decay of the synapse
      # I(x,t) is the input at position x and time t
      # Ω is the domain of integration of size lxl (m²)

Simulation parameters:
      # n  : space discretisation
      # dt : temporal discretisation (s)
      # t  : duration of the simulation (s)

The integration is made over the finite 2d domain [-l/2,+l/2]x[-l/2,+l/2]
discretized into n x n elements considered as a toric surface, during a period
of t seconds.
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2,ifft2,fftshift,ifftshift


def disc(shape=(256,256), center=None, radius = 64):
    ''' Generate a numpy array containing a disc.

    :Parameters:
        `shape` : (int,int)
            Shape of the output array
        `center`: (int,int)
            Disc center
         `radius`: int
             Disc radius (if radius = 0 -> disc is 1 point)
    '''
    if not center:
        center = (shape[0]//2,shape[1]//2)    
    def distance(x,y):
        return np.sqrt((x-center[0])**2+(y-center[1])**2)
    D = np.fromfunction(distance,shape)
    return np.where(D<=radius,True,False).astype(np.float32)


def peel(Z, center=None, r=8):
    ''' Peel an array Z into several 'onion rings' of width r.

    :Parameters:
        `Z`: numpy.ndarray
            Array to be peeled
        `center`: (int,int)
            Center of the 'onion'
        `r` : int
            ring radius
    :Returns:
        `out` : [numpy.ndarray,...]
            List of n Z-onion rings with n ≥ 1
    '''
    if r <= 0 :
        raise exceptions.ValueError('Radius must be > 0')
    if not center:
        center = (Z.shape[0]//2,Z.shape[1]//2)
    if  (center[0] >= Z.shape[0] or center[1] >= Z.shape[1] or \
        center[0] < 0 or center[1] < 0 ) : 
        raise exceptions.ValueError('Center must be in the matrix')

    # Compute the maximum diameter to get number of rings
    dx = float(max(Z.shape[0]-center[0],center[0]))
    dy = float(max(Z.shape[1]-center[1],center[1]))
    radius = np.sqrt(dx**2+dy**2)

    # Generate 1+int(d/r) rings
    L = []
    K = Z.copy()
    n = 1+int(radius/r)
    for i in range(n):
        r1 = (i  )*r/2
        r2 = (i+1)*r/2
        K = (disc(Z.shape,center,2*r2) - disc(Z.shape,center,2*r1))*Z
        L.append(K)
    L[0][center[0],center[1]] = Z[center[0],center[1]]
    return L


# -----------------------------------------------------------------------------
def gaussian(x, sigma=1.0):
    ''' Gaussian function of the form exp(-x²/2σ²)/(2π.σ²) '''
    return np.exp(-x**2/(2.0*sigma**2))/(2.0*np.pi*sigma**2)

def g(x, sigma=1.0):
    ''' Gaussian function of the form exp(-x²/2σ²)) '''
    return np.exp(-0.5*(x/sigma)**2)

def sigmoid(x):
    ''' Sigmoid function of the form 1/(1+exp(-x)) '''
    return 1.0/(1+np.exp(-x))



if __name__ == '__main__':
    import sys
    
    # Parameters
    # ----------
    l     = 10.00  # size of the field (mm)
    n     = 256    # space discretization
    c     = 10.0   # velocity of an action potential (m/s)
    dt    = 0.010  # temporal discretisation (in seconds)
    tau   = 1.0    # temporal decay of the synapse

    # Kernel function
    phi_0 = 0*np.pi/3.0
    phi_1 = 1*np.pi/3.0
    phi_2 = 2*np.pi/3.0
    K0    = 0.1
    k_c   = 10*np.pi/l
    sigma = 10
    K_xy = '''K0*(np.cos(k_c*(x*np.cos(phi_0)+y*np.sin(phi_0))) + \
                  np.cos(k_c*(x*np.cos(phi_1)+y*np.sin(phi_1))) + \
                  np.cos(k_c*(x*np.cos(phi_2)+y*np.sin(phi_2)))) * \
                  np.exp(-np.sqrt(x*x+y*y)/sigma)'''

    # Firing rate function
    S_v = '2.0/(1.0+np.exp(-5.5*(v-3)))'

    # Kernel initialisation
    # ---------------------
    # Generate kernel K from kernel definition Kd
    #  Kd is a string function of d where d represents distance between neurons
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    y_inf, y_sup, cy, dy = -l/2, +l/2, 0, l/float(n)
    nx, ny = (x_sup-x_inf)/dx, (y_sup-y_inf)/dy
    X,Y = np.meshgrid(np.arange(x_inf,x_sup,dx), np.arange(y_inf,y_sup,dy))
    D = np.sqrt(X**2+Y**2)
    K = eval(K_xy, globals(), {'x':X, 'y':Y})*dx*dy

    # Input
    I = 2
    I0 = 1.0
    sigma_i = 0.2
    I_ext = I0*gaussian(D,sigma_i)

    # Initial state (t ≤ 0)
    V0 = 2.00083
    V = np.ones((n,n))*V0

    # Here we generate kernel rings
    # The theoretical ring width value is r = c*l*dt*n and we need an integer
    # value (for precise computation). We thus compute the theoretical value,
    # take the integer part and modify dt such that it fits.
    # r = max(1,int(l/(c*dt)*n))
    # r = max(1, (np.sqrt(2)*n)/(l/(c*dt)))
    # dt = r/(l*n*c)
    r = c*dt*n/l
    Kl = peel(K, center=(n//2,n//2), r=r)
    nrings = len(Kl) # Number of rings
    
    # We precompute Fourier transform for each kernel ring since we'll only use
    # them in the Fourier domain
    # Kl = [fft2(Kl[i]) for i in range(nrings)]
    Kl = [fft2(fftshift(Kl[i])) for i in range(nrings)]

    # Print parameters
    # ----------------
    print '---------------------'
    print 'Simulation parameters'
    print '---------------------'
    print 'Size of the field        : %.2fmm×%.2fmm' % (l,l)
    print 'Action potential velocity: %.2fmm/s' % c
    print 'Kernel function          : %s' % K_xy
    print 'Firing rate function     : %s' % S_v
    print 'Tau                      : %f' % tau
    print 'Space discretisation     : %d×%d' % (n,n)
    print 'Time discretisation      : %.2fms' % (dt)
    print 'Number of rings          : %d' % nrings
    print 'K sum                    : %f' % K.sum()

    # Initialisation
    # ---------------
    # Initialisation of past S(V) values (from t=-Tmax to t=0, where Tmax = nrings*dt)
    # Since we're working in the Fourier domain, past values are directly stored using
    # their Fourier transform
    S = [fft2(eval(S_v, globals(), {'v' : V}))]*(nrings)

    # Run simulation
    # --------------
    import glumpy
    window = glumpy.Window(512, 512)
    V_float = np.zeros((n,n),dtype=np.float32)
    Im = glumpy.Image(V_float, interpolation='nearest',
                      cmap=glumpy.colormap.Grey, vmin= 2.000, vmax=2.5)

    @window.event
    def on_draw():
        window.clear()
        Im.blit(0,0,512,512)
    @window.event
    def on_key_press(key, modifiers):
        if key == glumpy.key.ESCAPE:
            sys.exit()
    t = 0
    @window.event
    def on_idle(*args):
        global V, S, Kl, I, t, dt, S_v, n

        print 'Time %.3fms:' % t
        print '   V_min = %.8f' % V.min()
        print '   V_max = %.8f' % V.max()
        
        L = Kl[0]*S[0]
        for j in range(1,nrings):
            L += Kl[j]*S[j]
        L = ifft2(L).real
        if (t < 60*dt):
            dV = dt/tau*(-V+L+I)
        else:
            dV = dt/tau*(-V+L+I+I_ext)
        V += dV
        S = [fft2(eval(S_v, globals(), {'v' : V})),] + S[:-1]

        t += dt
        V_float[...] = V
        Im.update()
        window.draw()

    window.mainloop()
