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
import numpy

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
    # a live cell is killed if it has fewer 
    # than 2 or more than 3 neighbours.
    part1 = ((Z == 1) & (N < 4) & (N > 1)) 
    # a new cell forms if a square has exactly three members
    part2 = ((Z == 0) & (N == 3))
    return (part1 | part2).astype(int)

Z = numpy.array([[0,0,0,0,0,0],
                 [0,0,0,1,0,0],
                 [0,1,0,1,0,0],
                 [0,0,1,1,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0]])

print 'Initial state:\n', Z[1:-1,1:-1]
for i in range(4): Z = iterate(Z)
print 'Final state:\n', Z[1:-1,1:-1]
