.. currentmodule:: dana

===============================================================================
From numpy array to dana group                                                 
===============================================================================

.. only:: html

   .. contents::
      :local:
      :depth: 2


For brevity and convenience, we assume from now on that main packages (dana,
numpy, scipy, and matplotlib) have been imported as::

   >>> import dana
   >>> import numpy as np
   >>> import matplotlib as mpl
   >>> import matplotlib.pyplot as plt

An alternative way is to write::

   >>> from dana import *

that will makes numpy, numpy.random and matpotlib.pyplot available as
``np``, ``rnd`` and ``plt`` and dana objects directly accessibles.


Data structures                                                                
===============================================================================


Memory layout                                                                  
-------------------------------------------------------------------------------
One the main object  of dana is the :class:`Group` class which  is more or less
equivalent to a numpy structured array with some subtle differences from a user
point of view. Numpy  array layout is such that the data  pointer points to one
block of N items, where each item is described by the dtype::

   >>> Z = np.zeros((3,3), [('x',float), ('y',float)])
   >>> print Z['x'].flags['CONTIGUOUS']
   False

DANA group layout is quite different since each element of the dtype is made of
a contiguous block of memory::

   >>> Z = dana.zeros((3,3), [('x',float), ('y',float)])
   >>> print Z['x'].flags['CONTIGUOUS']
   True

This is  a design choice to  make things a  little bit faster since  each group
field may  be subject to intensitve  computation and the  interleaved nature of
numpy structured  arrays would  slow down  things. This means  that a  group is
*not*  a numpy  array. Consequently,  even if  dana tries  to ensure  a maximum
compatibility  between  the  two  of   them,  there  may  be  nonetheless  some
incompatibilities or missing features.

Conversion                                                                     
-------------------------------------------------------------------------------
.. warning::

   Because of the fundamental difference in respective memory layouts, each
   conversion implies a whole copy of the data.

Conversion from group to array and vice-versa is straightforward:

Group to array::

   >>> G = zeros((3,3), [('x',float), ('y',float)])
   >>> A = G.asarray()

Array to group::

   >>> A = np.zeros((3,3), [('x',float), ('y',float)])
   >>> G = Group(A)


Field access                                                                   
-------------------------------------------------------------------------------
A specific field of a group can be accessed using three different syntaxes:

Accessing field as a regular attribute (the result is the underlying numpy
array representing the requested field)::

   >>> G = zeros((3,3), [('x',float), ('y',float)])
   >>> print type(G.x)
   <type 'numpy.ndarray'>

Accessing field as an item (the result is the underlying numpy array
representing the requested field)::

   >>> print type(G['x'])
   <type 'numpy.ndarray'>


Accessing field through a function call::

   >>> print type(G('x'))
   <class 'dana.group'>

The result is a new group with a unique field corresponding to the one
requested. Note that this group is only a placeholder since the actual
underlying numpy array is not copied::

   >>> G('x').x is G.x
   True


The Game of Life                                                               
===============================================================================

Even if dana is slanted toward computational neuroscience, we'll consider in
this section the `game of life
<http://en.wikipedia.org/wiki/Conway's_Game_of_Life>`_ by John Conway which is
one of the earliest example of cellular automata (see figure below). Those
cellular automaton can be conveninetly considered as groups of units that are
connected together through the notion of neightbours.  We'll show in the
following sections implementation of this game using pure python, numpy and
dana in order to illustrate main concepts of dana as well as main differences
with python and numpy.

.. figure:: _static/game-of-life.png

   **Figure 1** Simulation of the game of life.




Definition                                                                     
-------------------------------------------------------------------------------
.. note:: 

   This is an excerpt from `wikipedia
   <http://en.wikipedia.org/wiki/Cellular_automaton>`_ entry on Cellular
   Automaton.

*The Game of Life, also known simply as Life, is a cellular automaton devised
by the British mathematician John Horton Conway in 1970.  It is the
best-known example of a cellular automaton. The "game" is actually a
zero-player game, meaning that its evolution is determined by its initial
state, needing no input from human players. One interacts with the Game of
Life by creating an initial configuration and observing how it evolves.*

*The universe of the Game of Life is an infinite two-dimensional orthogonal grid
of square cells, each of which is in one of two possible states, live or
dead. Every cell interacts with its eight neighbours, which are the cells that
are directly horizontally, vertically, or diagonally adjacent. At each step in
time, the following transitions occur:*

- *Any live cell with fewer than two live neighbours dies, as if by needs
  caused by underpopulation.*
- *Any live cell with more than three live neighbours dies, as if by
  overcrowding.*
- *Any live cell with two or three live neighbours lives, unchanged, to the
  next generation.*
- *Any dead cell with exactly three live neighbours becomes a live cell.*

*The initial pattern constitutes the 'seed' of the system.  The first generation
is created by applying the above rules simultaneously to every cell in the seed
– births and deaths happen simultaneously, and the discrete moment at which
this happens is sometimes called a tick. (In other words, each generation is a
pure function of the one before.)  The rules continue to be applied repeatedly
to create further generations.*


The way of python                                                              
-------------------------------------------------------------------------------
In pure python, we can code the Game of Life using a list of lists representing
the board where cells are supposed to evolve::

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
the Game of Life rules::

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



The way of numpy                                                               
-------------------------------------------------------------------------------
Using numpy, we can benefit from vectorized computation and accelerates things
a lot. The board can now be represented using a numpy array::

   >>> Z = np.array ([[0,0,0,0,0,0],
                      [0,0,0,1,0,0],
                      [0,1,0,1,0,0],
                      [0,0,1,1,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0]])

This board possesses a ``0`` border that allows to accelerate things a bit by
avoiding to have specific tests for borders when counting the number of
neighbours. To iterate one step in time, we count the number of neighbours for
all internal cells at once and we update the whole board according to the Game
of Life rules::

   def iterate(Z):
       # find number of neighbours that each square has
       N = np.zeros(Z.shape)
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


The way of dana                                                                
-------------------------------------------------------------------------------
As for numpy, the first things to do is to create a :class:`Group` for holding
our cells. Howver, instead of simply declaring group dtype, we can directly give
the equation governing each value such that:

* *V* represents the current state of a cell
* *N* is the number of active neighbours

According to the game of life rules, we know that:

* every cell with less than 1 neighbours must die: (N<1.5)
* every cell with more than 4 neighbours must die: (N>3.5)
* every dead cell with less than 3 neighbours must die: (N<2.5)*(1-state)
* every other cell remains unchanged.

Thus, we declare Z as::

   >>> Z = Group((4,4), '''V = maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V)) : int
                           N : int''')

and we initialize the *V* value with the glider pattern::

   >>> Z.V = np.array([[0,0,1,0],
                       [1,0,1,0],
                       [0,1,1,0],
                       [0,0,0,0]])

This group is now made of 4x4 cells, each of them having a two values named *V*
and *N*. The first value *V* has been specified using an :class:`Equation`
while the second is a simple :class:`Declaration` that will be a placeholder
for the connection output.

Each cell now needs to be connected to its immediate neighbours and this can be
done by using a :class:`Connection` to connect *Z* to itself (see
chapter :doc:`connection` for further details)::
 
   >>> C = SharedConnection(Z('V'), Z('N'),
                            np.array([[1., 1., 1.], 
                                      [1., 0., 1.], 
                                      [1., 1., 1.]]))

Cells are now linked to their immediate neighbours using a (shared) connection
that will output in the *N* field in *Z*. This connection represents the
weighted sum of cell *state* activity using given array.  Since array values
are either 0 or 1 and cell states are either 0 or 1 , the weighted sum actually
represents the number of live cells in the immediate neighboorhood.
