.. title:: Tutorial
.. include:: common.rst

Tutorial
========

As an introduction, we will implement the well known `game of life`_ by John
Conway which is (according to wikipedia) a finite two-dimensional grid of
square cells, each of which is in one of two possible states, *live* or *dead*.
Every cell interacts with its eight neighbours, which are the cells that are
directly horizontally, vertically, or diagonally adjacent. At each step in
time, the following transitions occur:

1. Any live cell with fewer than two live neighbours dies, as if caused by
   underpopulation.
2. Any live cell with more than three live neighbours dies, as if by
   overcrowding.
3. Any live cell with two or three live neighbours lives on to the next
   generation.
4. Any dead cell with exactly three live neighbours becomes a live cell.


Implementation
--------------

The first thing to do is then to create a finite grid for holding our
cells. This corresponds to the notion of a group within dana:

.. code-block:: python

   >>> import dana
   >>> G = dana.zeros((50,50), dtype=int)

``G`` is now a group made of 50x50 cells, each of them having a single integer
value named ``V`` (this is the default name). Each cell needs now to be
connected to its immediate neighbours.  This can be done by first creating a
connection kernel that will be used to connect ``G`` to itself:

.. code-block:: python

   >>> import numpy
   >>> K = numpy.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
   >>> G.connect(G.V, K, 'N'

``G`` now  possesses a connection named ``N``  that links each cell  to its eight
neighbours:

.. code-block:: python

   >>> print G.N[1,1]
   array([[  1.,   1.,   1., ...,  NaN,  NaN,  NaN],
          [  1.,  NaN,   1., ...,  NaN,  NaN,  NaN],
          [  1.,   1.,   1., ...,  NaN,  NaN,  NaN],
          ..., 
          [ NaN,  NaN,  NaN, ...,  NaN,  NaN,  NaN],
          [ NaN,  NaN,  NaN, ...,  NaN,  NaN,  NaN],
          [ NaN,  NaN,  NaN, ...,  NaN,  NaN,  NaN]])

This connection is accessible within the equation of ``V`` as ``N`` and
represents the weighted sum of the kernel and the corresponding cell values.
Given ``K``, this weighted sum actually represents the number of active
neighbours and we can now define the equation of ``V`` over time given that:

    a. ``V`` is the the current state
    b. ``N`` is the number fo active neighbours

We can thus compute cells who need to die according to the rules:

    1. Every cell with less than 1 neighbours dies: (N<1.5)
    2. Every cell with more than 4 neighbours dies: (N>3.5)
    3. Every dead cell with less than 3 neighbours dies: (N<2.5)*(1-V)
    4. Every other cell lives or is made alive.


This can be written as:

.. code-block:: python

   >>> G.dV = '-V + maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'

where ``G.dV`` designates the evolution of V over time, i.e. G.V(t+dt) =
G.V(t) + dV(t).



Simulation
----------

At this point, we can now randomly initialize ``G.V`` and run some iterations.

.. code-block:: python

   >>> G.V = numpy.random.randint(0,2,G.shape)
   >>> for i in range(50):
   >>>     G.compute()

And we display the result using matplotlib_.

.. code-block:: python

   >>> import matplotlib.pyplot as plt
   >>> plt.imshow(G.V, cmap=pylab.cm.gray_r, vmin=0, vmax=1)
   >>> plt.xticks([])
   >>> plt.yticks('[])
   >>> plr.show()
