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
# $Id$
#------------------------------------------------------------------------------

# Import 
import dana.core as core
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile


def make_map (P, size):
    m = core.Map ( (size, size), (0,0) )
    l = core.Layer()
    m.append (l)
    l.fill (core.Unit)

    proj          = projection.projection()
    proj.self     = True
    proj.distance = distance.euclidean (False)
    proj.density  = density.full (1)
    proj.shape    = shape.box (.25, .25)
    proj.profile  = profile.gaussian (1, .25)
    proj.src = l
    proj.dst = l
    P.append(proj)
    
    return m

if __name__ == '__main__':
    import time

    size = 60
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


    
    
