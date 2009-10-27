
=============
Link equation
=============

Link equation obeys the same rule as :ref:`value equation <equation-value>` but
more variables are accessible and naming convention is different:

* Any pre-synaptic activity *V* from the source group is accesible via the naming
  convention ``pre['V']``

* Any post-synaptic activity *V* from the target source is accesible via the naming
  convention ``post['V']``

* Actual link value is named 'W' to differentiate it from the computed link
  value. Note that computed link value is not accessible if no value computation
  has been performed:

.. code-block:: python

   >>> G1 = dana.ones((3,3))
   >>> G2 = dana.ones((3,3))
   >>> G1.connect(G2, 'I', numpy.ones((1,1)), shared=False)
   >>> G1.equation['V'] = 'I'
   >>> G1.equation['I'] = "pre['V']*W - post['I']"

In the left part of the last line of the above example, *W* refers to the actual
link value (kernel elements) while *I* refers to the computed link value.


.. code-block:: python

   >>> G1.compute(dt=0.1) # I is now accessible
   >>> G1.learn(dt=0.1)
   >>> print G1.get_weight(S,0,0)

