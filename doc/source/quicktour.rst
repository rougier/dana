.. _quick-tour:
.. highlight:: none

===============================================================================
Quick tour                                                                     
===============================================================================

:class:`~dana.Equation` and :class:`~dana.DifferentialEquation` allows to
respectively compute equations and first order differential equations.
:class:`~dana.Group` is more or less equivalent to a numpy array with some
differences and the extra possibility of specifying a model which is a set of
equations. :class:`~dana.Connection` (:class:`~dana.DenseConnection`,
:class:`~dana.SparseConnection`, :class:`~dana.SharedConnection`) allows to
compute the weighted sum of a numpy array (or a dana group) using a given
kernel.

**Equations**::

   >>> eq = dana.Equation('y = a+b')
   >>> print eq(y=1, a=1, b=1)
   2
   >>> print eq(y=numpy.ones((2,2)), a=1, b=1)
   [[2. 2.]
    [2. 2.]]

**Differential Equations**::

   >>> eq = dana.DifferentialEquation('dy/dt = y')
   >>> print eq.run(y=1, t=1.0, dt=0.01)

**Groups**::

   >>> G = dana.ones(1, dtype=[('x' ,float),('y' ,float)])
   >>> 
   >>> G = dana.ones(1, model='dV/dt = (V+1)')
   >>> G.run(t=1.0,dt=0.01)
   >>> print G
   [ ]

**Connections**::

   >>> Z = numpy.ones((3,3))
   >>> K = numpy.ones((3,3))
   >>> print dana.DenseConnection(Z,Z,K).evaluate()
   [[4., 6., 4.]
    [6., 9., 6.]
    [4., 6., 4.]]

**Putting all together**::

   src = dana.ones((50,50), '''dV/dt = 0.1*N : float
                               N : float''')
   K = numpy.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]]
   SparseConnection(src('V'), src('N'), K))
   run(t=10.0, dt=0.1)
