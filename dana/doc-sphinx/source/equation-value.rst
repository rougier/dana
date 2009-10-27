.. _equation-value:

==============
Value equation
==============

When defining a group value equation, several variables are accessible:

* Any group value
* Any group constants (that need to be defined before evaluation)
* Any computed link value held by the group
* *dt* as given by the ``compute`` method.

.. code-block:: python

   >>> G1 = dana.ones((3,3))
   >>> G2 = dana.ones((3,3))
   >>> G1.connect(G2, 'I', numpy.ones((1,1)), shared=False)
   >>> G1.constant['k'] = .03
   >>> G1.equation['V'] = 'V+I*dt + k'
   >>> G1.compute(dt=0.1)
   0.9
   >>> print G1
   [[(1.13, True) (1.13, True) (1.13, True)]
    [(1.13, True) (1.13, True) (1.13, True)]
    [(1.13, True) (1.13, True) (1.13, True)]]

The value of ``I`` in the case above is a weighted sum, this the default
computation that is performed for any link when no further specifications are
given. Some other computations are also available and can be specified when
connecting a group to another:

.. code-block:: python

   >>> # Weighted sum
   >>> G1.connect(G2, 'I*', numpy.ones((1,1)))
   >>> # Distance
   >>> G1.connect(G2, 'I-', numpy.ones((1,1)))

