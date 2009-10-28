
Game of Life                                                                   
===============================================================================

.. contents::
   :depth: 1
   :local:

Theory                                                                         
-------------------------------------------------------------------------------

**Text from wikipedia entry** [1]_

.. figure:: _static/game-of-life.png
   :figclass: figure-right

   40x40 cells arena after 50 iterations.

The Game of Life, also known simply as Life, is a cellular automaton devised
by the British mathematician John Horton  Conway in 1970.  It is the best-known
example of  a cellular  automaton. The "game"  is actually a  zero-player game,
meaning that its evolution is determined by its initial state, needing no input
from human players. One interacts with  the Game of Life by creating an initial
configuration and observing how it evolves.

The universe of the Game of Life is an infinite two-dimensional orthogonal grid
of  square cells,  each of  which is  in one  of two  possible states,  live or
dead. Every cell interacts with its  eight neighbours, which are the cells that
are directly horizontally, vertically, or  diagonally adjacent. At each step in
time, the following transitions occur:

    1. Any live cell with fewer than two live neighbours dies, as if by needs
       caused by underpopulation.
    2. Any live cell with more than three live neighbours dies, as if by
       overcrowding.
    3. Any live cell with two or three live neighbours lives, unchanged, to the
       next generation.
    4. Any dead cell with exactly three live neighbours becomes a live cell.

The initial pattern constitutes the 'seed' of the system.  The first generation
is created by applying the above rules simultaneously to every cell in the seed
-- births and  deaths happen simultaneously,  and the discrete moment  at which
this happens is sometimes called a  tick. (In other words, each generation is a
pure function of the one before.)   The rules continue to be applied repeatedly
to create further generations.

Implementation
++++++++++++++

The first thing to do to simulate the game of life is to create a 2d arena with
a given size (40x40 in this case).

.. code-block:: python

   import numpy, dana
   arena = dana.group((40,40), name='arena')

Since we did not specified any field for the group, it will possess the default
*V* value as well  as the *mask* value that record dead  cells. Once this arena
has been created, we also need to  connect every cell to its 8 neighbours using
the proper kernel.

.. code-block:: python

   K = numpy.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
   arena.connect(arena['V'], 'N', K, shared=True)

This connection is named *N* and will be accessible within the equation of *V*.
This *N* value represents the weighted  sum of the kernel and the corresponding
cell values.   Given *K*, this weighted  sum actually represents  the number of
active neighbours.

Now, we need to define the equation of value *V* over time given:

    a. *V* the current state of the arena
    b. *N* the number fo active neighbours

To do so, we will compute cells who need to die according to the rules:

    1. Every cell with less than 1 neighbours: (N<1.5)
    2. Every cell with more than 4 neighbours: (N<3.5)
    3. Every dead cell with less than 3 neighbours: (N<2.5)*(1-V)


.. code-block:: python

   arena.equation['V'] = 'maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'

Finally, we randomly initialize the arena and compute some iterations.

.. code-block:: python

   arena['V'] = numpy.random.randint(0,2,arena.shape)
   for i in range(50):
       arena.compute()


And we display the result using pylab.

.. code-block:: python

   view = dana.pylab.view(arena['V'], cmap=pylab.cm.gray_r, vmin=0, vmax=1)
   pylab.xticks('')
   pylab.yticks('')
   view.show()


----------

*Notes*

    .. [1] http://en.wikipedia.org/wiki/Conway's_Game_of_Life
