The Game of Life, the dana way
==============================

The first thing to do is to create a group for holding our cells:

.. code-block:: python

   >>> import numpy, dana
   >>> Z = dana.group([[0,0,1,0],
                       [1,0,1,0],
                       [0,1,1,0],
                       [0,0,0,0]])

This group is now made of 4x4 cells, each of them having a single integer value
named *V* (this is the default name).  Each cell needs to be connected to its
immediate neighbours. This can be done by using a connection kernel to connect
*Z* to itself:

.. code-block:: python

   >>> Z.connect(Z.V, numpy.array([[1,1,1],
                                   [1,0,1],
                                   [1,1,1]]), 'N')

Cells are now linked to their immediate neighbours using a connection named
*N*. This connection represents the weighted sum of cell *state* activity and
weight links.  Since link values are either 0 or 1 (because the kernel is made
of 0 and 1) and cell states are either 0 or 1 , the weighted sum actually
represents the number of live cells in the immediate neighboorhood. Using this
information, we can now define how the *state* variable evolve between time *t*
and *t+dt* given that:

* *V* represents the current state of a cell
* *N* is the number of active neighbours

According to the game of life rules, we know that:

* every cell with less than 1 neighbours must die: (N<1.5)
* every cell with more than 4 neighbours must die: (N>3.5)
* every dead cell with less than 3 neighbours must die: (N<2.5)*(1-state)
* every other cell remains unchanged.

Thus, we can write the *dV* equation for the group Z:

.. code-block:: python

   >>> Z.dV = '-V + maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'


Finally, we iterate 4 steps in time and we check the `glider
<http://en.wikipedia.org/wiki/Glider_(Conway's_Life)>`_ has glided one step
down and right:

.. code-block:: pycon

   >>> print Z
   [[0 0 1 0]
    [1 0 1 0]
    [0 1 1 0]
    [0 0 0 0]]
   >>> for i in range(4):
           Z.compute()
   >>> print Z
   [[0 0 0 0]
    [0 0 0 1]
    [0 1 0 1]
    [0 0 1 1]]


Sources
-------

`game-of-life_dana.py <_static/game-of-life_dana.py>`_
