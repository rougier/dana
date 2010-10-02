.. include:: header-examples.txt

Game of Life
============

.. code-block:: python

   from dana import *

   src = Group((50,100),
                '''V = maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V)) : int
                   N : float''')
   C = SharedConnection(src('V'), src('N'), np.array([[1, 1, 1], 
                                                      [1, 0, 1], 
                                                      [1, 1, 1]]))
   src.V = rnd.randint(0, 2, src.shape)
   run(n=100)

|

Heat diffusion
==============

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

.. code-block:: python

   from dana import *

   n = 40
   p = 2*n+1
   alpha, tau, h = 1.0, 0.1, 0

   input = np.zeros((n,n))
   focus = Group((n,n), '''dU/dt = alpha*(-V + tau*(L+I)) +h : float
                            V    = np.maximum(U,0)           : float
                            I                                : float
                            L                                : float''')
   SparseConnection(input, focus('I'), np.ones((1,1)))
   SharedConnection(focus('V'), focus('L'),
                    1.25*gaussian((p,p),0.1) - 0.75*gaussian((p,p),1.0))
   input[...] = gaussian((n,n),0.25,(0.5,0.5))   \
              + gaussian((n,n),0.25,(-0.5,-0.5)) \
              + (2*rnd.random((n,n))-1)*.05
   run(t=5.0, dt=0.01)
