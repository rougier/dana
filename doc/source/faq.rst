.. _faq:
.. highlight:: none

==========================
Frequently asked questions
==========================

What is DANA ?
==============
Please have a look at " :doc:`intro` " section of the documentation.


Differential equation does not update my variable
=================================================
Let us consider the following code::

   >>> y = 1
   >>> eq = dana.DifferentialEquation('dy/dt = y')
   >>> eq.evaluate(y, dt=0.1)
   1.1
   >>> print y
   1

The reason the ``y`` variable is not updated is that you pass ``y`` as a
parameter for the ``evaluate`` function and it is thus not updated in the main
namespace. If you want to ``y`` to be updated, you will have to write::

   >>> y = 1
   >>> eq = dana.DifferentialEquation('dy/dt = y')
   >>> y = eq.evaluate(y, dt=0.1)
   1.1
   >>> print y
   1.1


Connection does not propagate to target
=======================================
Let us consider the following code::

  >>> S = np.ones((1,))
  >>> T = np.zeros((1,))
  >>> C = dana.DenseConnection(S,T,np.ones((1,))
  >>> C.propagate()
  >>> print T
  [ 1. ]
  >>> S = 2*np.ones((1,))  # Faulty line
  >>> C.propagate()
  >>> print T
  [ 1. ]

While the first propagation hold the expected result, the second one does not
work since you re-affect a new numpy array to the ``S`` variable while the
connection source is still the old ``S``. If you want to modify source, you
would have to write::

  >>> S = np.ones((1,))
  >>> T = np.zeros((1,))
  >>> C = dana.DenseConnection(S,T,np.ones((1,))
  >>> C.propagate()
  >>> print T
  [ 1. ]
  >>> S[...] = 2
  >>> C.propagate()
  >>> print T
  [ 2. ]


Models and/or groups evaluation is not consistent
=================================================
Let us consider the following example::

  >>> G = dana.Group((1,3), 'dV/dt = 1; U = U+V')
  >>> G[...] = (1,0)
  >>> G.run(1,0.1)
  >>> print G
  [ 2.0  15.5 ]
  >>> G[...] = (1,0)
  >>> G.run(1,0.01)
  >>> print G
  [ 2.0  150.5 ]

The ``V`` field of ``G`` is a differential equation that explicitely depends on
time while the ``U`` field is a simple equation that does not depend
explicitely depends on time. However, it does implicitely depends on time since
the equation is evaluated at each time step. In the first run, there are ten
iterations (t/dt) while in the second run, there are 100 iterations. This is
the reason why the results are different.
