The Game of Life, the python way
================================

In pure python, we can code the Game of Life using a list of lists 
representing the board where cells are supposed to evolve:

.. code-block:: pycon

   >>> Z = [[0,0,0,0,0,0],
            [0,0,0,1,0,0],
            [0,1,0,1,0,0],
            [0,0,1,1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]]

This board possesses a ``0`` border that allows to accelerate things a bit by
avoiding to have specific tests for borders when counting the number of
neighbours. To iterate one step in time, we simply count the number of
neighbours for each internal cell and we update the whole board according to
the Game of Life rules:

.. code-block:: python

   def iterate(Z):
       shape = len(Z), len(Z[0])
       N  = [[0,]*(shape[0]+2)  for i in range(shape[1]+2)]
       # Compute number of neighbours for each cell
       for x in range(1,shape[0]-1):
           for y in range(1,shape[1]-1):
               N[x][y] = Z[x-1][y-1]+Z[x][y-1]+Z[x+1][y-1] \
                       + Z[x-1][y]            +Z[x+1][y]   \
                       + Z[x-1][y+1]+Z[x][y+1]+Z[x+1][y+1]
       # Update cells
       for x in range(1,shape[0]-1):
           for y in range(1,shape[1]-1):
               if Z[x][y] == 0 and N[x][y] == 3:
                   Z[x][y] = 1
               elif Z[x][y] == 1 and not N[x][y] in [2,3]:
                   Z[x][y] = 0
       return Z

Finally, we iterate 4 steps in time and we check the `glider
<http://en.wikipedia.org/wiki/Glider_(Conway's_Life)>`_ has glided one step
down and right:

.. code-block:: pycon

   >>> display(Z)
   0 0 1 0
   1 0 1 0
   0 1 1 0
   0 0 0 0
   >>> for i in range(4):
           iterate(Z)
   >>> display(Z)
   0 0 0 0
   0 0 0 1
   0 1 0 1
   0 0 1 1


Sources
-------

`game-of-life_python.py <_static/game-of-life_python.py>`_
