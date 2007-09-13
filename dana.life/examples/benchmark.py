#!/usr/bin/env python
#
# DANA --- Game of Life
# Copyright (C) 2007 Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# $Id$


import random
import time
import dana.core as core
import dana.life as life
import dana.projection as proj


class Unit (core.Unit):
    """ Game of Life unit """

    def compute_dp (self):
        n = 0
        for src,w in self.laterals:
            n += src.potential
        if n in [3,4]:
            self._potential = 1.0
        else:
            self._potential = 0.0
        return self._potential - self.potential

    def compute_dw (self):
        self.potential = self._potential
        return 0.0

class MixedUnit (core.Unit):
    """ Mixed Game of Life unit """

    def compute_dp (self):
        n = self.compute_lateral_input()
        if n in [3,4]:
            self._potential = 1.0
        else:
            self._potential = 0.0
        return self._potential - self.potential

    def compute_dw (self):
        self.potential = self._potential
        return 0.0


# Create a new network
size  = 10

pynet = core.Network ()
pynet.append (core.Map ( (size,size), (0,0) ) )
pynet[0].append (core.Layer())
pynet[0][0].fill (Unit)

cpynet = core.Network ()
cpynet.append (core.Map ( (size,size), (0,0) ) )
cpynet[0].append (core.Layer())
cpynet[0][0].fill (MixedUnit)

cnet = core.Network ()
cnet.append (core.Map ( (size,size), (0,0) ) )
cnet[0].append (core.Layer())
cnet[0][0].fill (life.Unit)

p = proj.Projection()
p.self_connect = False
p.distance = proj.distance.Euclidean (False)
p.density  = proj.density.Full(1)
p.profile  = proj.profile.Constant(1.0)
p.shape    = proj.shape.Box (1.0/size, 1.0/size)

p.src = pynet[0][0]
p.dst = pynet[0][0]
p.connect()

p.src = cpynet[0][0]
p.dst = cpynet[0][0]
p.connect()

p.src = cnet[0][0]
p.dst = cnet[0][0]
p.connect()


epochs = 5000

print 'Running network with python units for %d epochs' % epochs
start = time.time()
for i in range(epochs):
    pynet.compute_dp()
    pynet.compute_dw()
print 'Elapsed time: %f seconds' % (time.time() - start)

print
print 'Running network with mixed C/python units for %d epochs' % epochs
start = time.time()
for i in range(epochs):
    cpynet.compute_dp()
    cpynet.compute_dw()
print 'Elapsed time: %f seconds' % (time.time() - start)

print
print 'Running network with C units for %d epochs' % epochs
start = time.time()
for i in range(epochs):
    cnet.compute_dp()
    cnet.compute_dw()
print 'Elapsed time: %f seconds' % (time.time() - start)
