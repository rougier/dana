#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
# 
# $Id$
#-------------------------------------------------------------------------------

import random
import dana.core as core
import dana.projection as proj


class Unit (core.Unit):
    """ """
    def compute_dp (self):
        n = 0
        for src,w in self.laterals:
            if src.potential == 1.0:
                n += 1
        if n == 3 or n == 4:
            self._potential = 1.0
        else:
            self._potential = 0.0
        return self._potential - self.potential

    def compute_dw (self):
        self.potential = self._potential
        return 0.0


# Create a new network
size  = 10
network = core.Network ()
map = core.Map ( (size,size), (0,0) )
network.append (map)
layer = core.Layer()
map.append (layer)
layer.fill (Unit)

# Connection to neighbours
p = proj.Projection()
p.self_connect = False
p.distance = proj.distance.Euclidean (False)
p.density  = proj.density.Full(1)
p.profile  = proj.profile.Constant(1.0)
p.shape    = proj.shape.Box (1.0/size, 1.0/size)
p.src = layer
p.dst = layer
p.connect()


for u in layer:
    u.potential = random.randint (0, 1)

print "Initial state:"
print layer.potentials()

n = 5
for i in range(n):
    print "Iteration %d" % i
    network.compute_dp()
    network.compute_dw()
    print layer.potentials()

