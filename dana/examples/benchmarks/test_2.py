#!/usr/bin/env python

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
    
    epochs = 5000
    print 'Running %d iterations' % epochs
    
    start = time.time()
    net.evaluate(epochs, False)
    end = time.time()
    print 'No thread version: %f second(s)' % (end-start)
    
    start = time.time()
    net.evaluate(epochs, True)
    end = time.time()
    print 'Threaded version:  %f second(s)' % (end-start)
    
