.. include:: header.incl
.. include:: footer.incl

===============================================================================
DANA                                                                           
===============================================================================
-------------------------------------------------------------------------------
Distributed (Asynchronous) Numerical & Adaptive computing framework            
-------------------------------------------------------------------------------

The Game of Life is a *game* designed by John Conway during the seventies. This
*game* is  actually a zero-player game  and all its evolution  is determined by
the  initial state.  One  can interact  with the  Game of  Life by  creating an
initial configuration and observing how it evolves.

.. figure:: game-of-life.png
   
   **Figure 1.** The universe of the  game is a two-dimensional board of square
   cells.  Each  state can  be either dead  (0) or  alive (1) depending  on its
   current state  and surrounding cells. **1.**  Any live cell  with fewer than
   two live  neighbours dies. **2.**  Any live cell  with more than  three live
   neighbours  dies. **3.** Any  live cell  with two  or three  live neighbours
   lives.  **4.** Any  dead cell  with  exactly three  live neighbours  becomes
   alive.

Implementing such a game using DANA is straightforward

.. code-block:: python

   import numpy, dana

   G = dana.zeros((50,100), dtype=int)
   K = numpy.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
   G.connect(G.V, K, 'N', sparse=True)
   G.dV = '-V+maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'
   G.V = numpy.random.randint(0,2,G.shape)
   for i in range(50):
       G.compute()
