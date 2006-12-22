#!/usr/bin/env python

# Import 
import dana.core as core
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile


def make_map (P, size):
    Map = core.Map ( (size, size), (0,0) )
    Layer = core.Layer()
    Map.append (Layer)
    Layer.fill (core.Unit)

    proj          = projection.projection()
    proj.self     = True
    proj.distance = distance.euclidean (False)
    proj.density  = density.full (1)
    proj.shape    = shape.box (.25, .25)
    proj.profile  = profile.gaussian (1, .25)
    proj.src = Layer
    proj.dst = Layer
    P.append(proj)
    
    return map

if __name__ == '__main__':
    import time

    size = 100
    P = projection.projector()    
    net = core.Network()
    
    net.clear()
    P.clear()
    for i in range(2):
        m = make_map (P, size)
        net.append(m)
        
    print 'No thread version :'
    print '-------------------'
    start = time.time()
    P.connect (False)
    end = time.time()
    print '%f second(s)' % (end-start)
    

    net.clear()
    P.clear()
    for i in range(2):
        m = make_map (P, size)
        net.append(m)

    print 'Threaded version :'
    print '------------------'
    start = time.time()
    P.connect (True)
    end = time.time()
    print '%f second(s)' % (end-start)


    
    
