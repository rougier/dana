Quick tour
===============================================================================

As   a   quick   tour   of   dana,   we'll  implement   the   `game   of   life
<http://en.wikipedia.org/wiki/Conway's_Game_of_Life>`_  by  John  Conway.   The
first thing to do is to create a group for holding our cells:

.. code-block:: python

   >>> import numpy
   >>> import dana
   >>> G = dana.zeros((40,40), dtype=bool)

This group  is now made of  40x40 cells, each  of them having a  single boolean
value named *V* (this is the default name).  Each cell needs to be connected to
its  immediate neighbours.  This can  be done  by first  creating  a connection
kernel that will be used to connect *G* to itself:

.. code-block:: python

   >>> K = numpy.array([[1, 1, 1],
   ...                  [1, 0, 1],
   ...                  [1, 1, 1]])
   >>> G.connect(G, 'N', K, sparse=True)

Cells are now linked to their immediate neighbours using a connection named
*N*. This connection represents the weighted sum of cell *state* activity and
weight links.  Since link values are either 0 or 1 (because K is made of 0 and
1) and cell states are either 0 (False) or 1 (True), the weighted sum actually
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

We can now write the *V* equation for the group G:

.. code-block:: python

   >>> G.dV = 'maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'

Last step, we randomly initialize the group and run some iterations.

.. code-block:: python

   >>> G.V = numpy.random.randint(0,2,G.shape)
   >>> for i in range(50):
   ...     G.compute()

Using `matplotlib <http://matplotlib.sourceforge.net/>`_, we can finally
visualize the result:

.. code-block:: python

   >>> import pylab
   >>> pylab.imshow(G.V)
   >>> pylab.show()

.. figure:: _static/game-of-life.png

   **Figure 1.** Game of  Life, group activity after 50 iterations.
