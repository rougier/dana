Group
=====

A group is a multi-dimensional array whose data type may be composed of one to
several values. In this sense, groups are very similar to numpy structured (or
record) arrays with one major difference though, each subfield of a group is a
contiguous numpy array:

.. code-block:: pycon

   >>> G = np.zeros((5,5), dtype=[('r',int),('g',int),('b',int)])
   >>> print G
   array([[(0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (0, 0, 0)]], 
         dtype=[('r', '<i8'), ('g', '<i8'), ('b', '<i8')])
   >>> print G['r'].flags['CONTIGUOUS']
   False
   >>>
   >>> G = dana.zeros((5,5), dtype=[('r',int),('g',int),('b',int)])
   >>> print G
   group([[(0, 0, 0, True), (0, 0, 0, True), (0, 0, 0, True)],
          [(0, 0, 0, True), (0, 0, 0, True), (0, 0, 0, True)],
          [(0, 0, 0, True), (0, 0, 0, True), (0, 0, 0, True)]], 
         dtype=[('r', '<i8'), ('g', '<i8'), ('b', '<i8'), ('mask', '|b1')])
   >>> print G['r'].flags['CONTIGUOUS']
   True


Sections
--------

.. toctree::
   :maxdepth: 1

   basics-group-creation.rst
   basics-group-manipulation.rst
