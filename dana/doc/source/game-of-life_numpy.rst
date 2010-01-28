The Game of Life, the numpy way
================================

Using numpy, we can benefit from vectorized computation and accelerates things
a lot. The board can now be represented using a numpy array:

.. code-block:: pycon

   >>> import numpy
   >>> Z = numpy.array ([[0,0,0,0,0,0],
                         [0,0,0,1,0,0],
                         [0,1,0,1,0,0],
                         [0,0,1,1,0,0],
                         [0,0,0,0,0,0],
                         [0,0,0,0,0,0]])

This board possesses a ``0`` border that allows to accelerate things a bit by
avoiding to have specific tests for borders when counting the number of
neighbours. To iterate one step in time, we count the number of neighbours for
all internal cells at once and we update the whole board according to the Game
of Life rules:

.. code-block:: python

   def iterate(Z):
       # find number of neighbours that each square has
       N = numpy.zeros(Z.shape)
       N[1:, 1:] += Z[:-1, :-1]
       N[1:, :-1] += Z[:-1, 1:]
       N[:-1, 1:] += Z[1:, :-1]
       N[:-1, :-1] += Z[1:, 1:]
       N[:-1, :] += Z[1:, :]
       N[1:, :] += Z[:-1, :]
       N[:, :-1] += Z[:, 1:]
       N[:, 1:] += Z[:, :-1]
       # a live cell is killed if it has fewer than 2 or more than 3 neighbours.
       part1 = ((Z == 1) & (N < 4) & (N > 1)) 
       # a new cell forms if a square has exactly three members
       part2 = ((Z == 0) & (N == 3))
       return (part1 | part2).astype(int)

Finally, we iterate 4 steps in time and we check the `glider
<http://en.wikipedia.org/wiki/Glider_(Conway's_Life)>`_ has glided one step
down and right:

.. code-block:: pycon

   >>> print Z[1:-1]
   [[0 0 1 0]
    [1 0 1 0]
    [0 1 1 0]
    [0 0 0 0]]
   >>> for i in range(4):
           iterate(Z)
   >>> print Z[1:-1]
   [[0 0 0 0]
    [0 0 0 1]
    [0 1 0 1]
    [0 0 1 1]]


Sources
-------

`game-of-life_numpy.py <_static/game-of-life_numpy.py>`_
