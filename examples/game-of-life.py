#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
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
            '''V = np.maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V)) : int
               N : float''')
C = SparseConnection(src('V'), src('N'), np.array([[1., 1., 1.], 
                                                   [1., 0., 1.], 
                                                   [1., 1., 1.]]))
src.V = rnd.randint(0, 2, src.shape)
run(n=100)

plt.figure(figsize=(8,4))
plt.imshow(1-src.V, interpolation='nearest',
           extent=[0,100,0,50], cmap=plt.cm.gray)
plt.xticks(range(0,101,10)), plt.yticks(range(0,51,10))
plt.grid()
plt.show()


