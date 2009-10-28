==================
Group manipulation
==================

A group may be thought as a set of numpy arrays and consequently, all operations
available on numpy arrays can be applied to any group value:

.. code-block:: python

   >>> G = dana.zeros((3,3))
   >>> G['V'] = cos(G['V'])
   >>> print G
   [[(1.0, True) (1.0, True) (1.0, True)]
    [(1.0, True) (1.0, True) (1.0, True)]
    [(1.0, True) (1.0, True) (1.0, True)]]

However, group-wide operation won't work as expected:

.. code-block:: python

   >>> G = dana.group((3,3))
   >>> print numpy.cos(G)
   NotImplemented

Most numpy `indexing and slicing operations
<http://docs.scipy.org/doc/numpy/user/basics.indexing.html>`_ work as usual on a
specific group value

.. code-block:: python

   >>> G = dana.group((10,10))
   >>> print G.shape
   (10, 10)
   >>> print G['V'][0,:]  = 1

while resizing and reshaping operations need to be applied group-wide.

.. code-block:: python

   >>> G = dana.group((10,10))
   >>> G.reshape(1,100)
   >>> G['V'].reshape(1,100)
   Traceback (most recent call last) :
   File "<stdin>", line 1, in <module>
   ValueError: resize only works on single-segment arrays
