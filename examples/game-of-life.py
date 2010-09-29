#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Implementation of Conway's Game of Life.

References:
-----------
  * M. Gardner, Scientific American, October 1970.
  * http://en.wikipedia.org/wiki/Conway's_Game_of_Life
'''
from dana import *

src = Group((50,100),
            '''V = maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V)) : int
               N : float''')
C = SharedConnection(src('V'), src('N'), np.array([[1., 1., 1.], 
                                                   [1., 0., 1.], 
                                                   [1., 1., 1.]]))
src.V = rnd.randint(0, 2, src.shape)
run(n=100)

plt.figure(figsize=(8,4))
plt.imshow(1-src.V, interpolation='nearest', extent=[0,100,0,50])
plt.xticks(range(0,101,10)), plt.yticks(range(0,51,10))
plt.grid()
plt.savefig("game-of-life.png",dpi=75)
plt.show()


