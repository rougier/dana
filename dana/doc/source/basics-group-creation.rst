==============
Group creation
==============

A group can be created using either one of the available basic creation routines

* :meth:`dana.empty` (shape, dtype, keys, mask, name)
* :meth:`dana.empty_like` (other)
* :meth:`dana.ones` (shape, dtype, keys, mask, name)
* :meth:`dana.ones_like` (other)
* :meth:`dana.zeros` (shape, dtype, keys, mask, name)
* :meth:`dana.zeros_like` (other)

or by using the group init function to specify all relevant parameters

* :meth:`dana.group.__init__` (shape, dtype, keys, mask, name, fill)

To specify the name and the type of the different values of the group, there is
two options.  If the values are homogeneous (same type), you can specify the
type using the ``dtype`` parameter and specify the name of each value using the
``keys`` parameter.

.. code-block:: python

   >>> G = dana.zeros((3,3), keys=['V'])
   >>> print G
   [[(0.0, True) (0.0, True) (0.0, True)]
    [(0.0, True) (0.0, True) (0.0, True)]
    [(0.0, True) (0.0, True) (0.0, True)]]

If you need non-homogeneous values, you have to specify name and type for each
value using only the ``dtype`` parameter (keys is not relevant in this case):

.. code-block:: python

   >>> G = dana.zeros((3,3), dtype=[('V',float), ('T', int)])
   >>> print G
   [[(0.0, 0, True) (0.0, 0, True) (0.0, 0, True)]
    [(0.0, 0, True) (0.0, 0, True) (0.0, 0, True)]
    [(0.0, 0, True) (0.0, 0, True) (0.0, 0, True)]]

Whatever the creation method that is used, a group always have a boolean
``mask`` value that indicates which element are active. If no keys is given and
the given dtype is a basic numpy type (see :ref:`table <numpy-type-table>`), a
default value named ``V`` is created.
