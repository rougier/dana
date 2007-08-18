#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: test_2.py 119 2007-02-07 14:16:22Z rougier $
#------------------------------------------------------------------------------

# Import
import math
import dana.core as core
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile


class Unit (core.Unit):
    def evaluate (self):
        return math.sqrt(2)


def make_map (size):
    Map = core.Map ( (size,size), (0,0) )
    Map.append(core.Layer())
    Map[0].fill (Unit)
    return Map

if __name__ == '__main__':
    import time

    # Create the network
    net = core.Network()
    for i in range(50):
        net.append (make_map(40))
    
    epochs = 1000
    print 'Running %d iterations' % epochs
    
    start = time.time()
    net.evaluate(epochs, False)
    end = time.time()
    print 'No thread version: %f second(s)' % (end-start)
    
    start = time.time()
    net.evaluate(epochs, True)
    end = time.time()
    print 'Threaded version:  %f second(s)' % (end-start)
    
