====
Link
====

A link represents a named connection between two groups (that can be possibly
the same). Since groups may have several values, a link is actually a named
connection between a group and a specific value of another group:

.. code-block:: python

   >>> S = dana.zeros((3,3), keys=['V'])
   >>> T = dana.zeros((3,3), keys=['V'])
   >>> T.connect(S['V'], 'I', numpy.ones((1,1)))
   >>> print T.get_weight(S,0,0)
   array([[  1.,  NaN,  NaN],
          [ NaN,  NaN,  NaN],
          [ NaN,  NaN,  NaN]])

When a group has only one value, it is possible to directly connect a group to
another without specifying the value to connect to since there is only one:

.. code-block:: python

   >>> T.connect(S, name='I', kernel=numpy.ones((1,1)))

In both cases, source group holds the link. This link corresponds to some
computation that will occur when the group is evaluated and will be stored into
the named variable (``I`` in the above example). See :ref:`equation` for more details on
the use of this variable.


.. toctree::
   :maxdepth: 1

   link-computation
   link-shared


