.. title:: Programming guide - DANA Basics
.. include:: common.rst

DANA basics
===========

At the core of the DANA package, is the group object that encapsulates a
special type of n-dimensional numpy record arrays. There are several important
differences between DANA groups and the standard numpy record array:

- DANA groups have a fixed size at creation, unlike Python lists (which can
  grow dynamically). Changing the size of a group will create a new array
  and delete the original.

- The elements in a DANA group are all required to be of the same data
  type, and thus will be the same size in memory.

- DANA group facilitate advanced mathematical and other types of
  operations on large numbers of data. Typically, such operations are
  executed more efficiently and with less code than is possible using
  Python's built-in sequences.


Vectorized computations
-----------------------

Data types
----------

Array creation
---------------

I/O with Numpy
--------------

Indexing
--------
