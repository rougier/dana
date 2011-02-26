.. include:: header-examples.txt

Game of Life
============

.. image:: images/game-of-life.png
   :class: left

.. code-block:: python

   from dana import *

   src = Group((50,100),
                '''V = maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V)) : int
                   N : float''')

   K =  np.array([[1, 1, 1], 
                  [1, 0, 1], 
                  [1, 1, 1]]))
   C = SharedConnection(src('V'), src('N'), K)

   src.V = rnd.randint(0, 2, src.shape)

   run(n=100)

|

Heat diffusion
==============

.. image:: images/diffusion.png
   :class: left

.. code-block:: python

   from dana import *

   n,k  = 40, .1
   src = Group((n,n), '''dV/dt = k*N : float
                         N           : float''')
   SparseConnection(src('V'), src('N'), np.array([[0, 1, 0], 
                                                  [1,-4, 1],
                                                  [0, 1, 0]]))
   src.V = 1

   for i in range(2500):
       src.run(dt=0.25)
       src.V[:,n-1] = src.V[n-1,:] = src.V[:,0] = 1
       src.V[0,:] = 0 

|

Sobel filter
============

.. image:: images/sobel.png
   :class: left

.. code-block:: python

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

|

Dynamic Neural Field
====================

.. image:: images/DNF.png
   :class: left

.. code-block:: python

   from dana import *

   n = 256
   src = np.zeros((n,))
   tgt = Group((n,), '''dU/dt = (-V + 0.1*L + I); V = maximum(U,0); I; L''')
   SparseConnection(src, tgt('I'), np.ones((1,)))
   SharedConnection(tgt('V'), tgt('L'), +1.10*gaussian(2*n+1, 0.20)
                                        -0.95*gaussian(2*n+1, 1.00))
   for i in range(10):
       a = 0.50*rnd.random(1)
       b = 0.25*rnd.random(1)
       c = 2.00*rnd.random(1)-1.0
       src[...] = np.maximum(src, a*gaussian(n, [b], [c]))
   run(t=30.0, dt=0.1)

|

Bellman-Ford algorithm
======================

.. image:: images/maze.png
   :class: left

.. code-block:: python

   from dana import *

   ...
   Z = 1-maze((n,n))
   G = Group((n,n),'''V = I*maximum(maximum(maximum(maximum(V,E),W),N),S)
                      W; E; N; S; I''')
   SparseConnection(Z,   G('I'), np.array([ [1] ]))
   SparseConnection(G.V, G('N'), np.array([ [a],      [np.NaN], [np.NaN] ]))
   SparseConnection(G.V, G('S'), np.array([ [np.NaN], [np.NaN], [a]      ]))
   SparseConnection(G.V, G('E'), np.array([ [np.NaN,  np.NaN,  a]        ]))
   SparseConnection(G.V, G('W'), np.array([ [a,       np.NaN,  np.NaN]   ]))
   G.V[-2,-1] = 1
   run(n=5*(n+n))
   ...

