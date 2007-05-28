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
# $Id: profile.py 119 2007-02-07 14:16:22Z rougier $
#------------------------------------------------------------------------------

import dana.core as core
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile

from dana.visualization.pylab import View

if __name__ == '__main__':

    size = 10;

    net = core.Network()
    
    m0 = core.Map( (size, size), (0,0) )
    m0.append( core.Layer() )
    m0[0].fill(core.Unit)
    m0.name = 'm0: constant'
    net.append(m0)
    
    m1 = core.Map( (size, size), (1,0) )
    m1.append( core.Layer() )
    m1[0].fill(core.Unit)
    m1.name = 'm1: linear'
    net.append(m1)
    
    proj          = projection.projection()
    proj.self     = True
    proj.distance = distance.euclidean(False)
    proj.density  = density.full(1)
    proj.shape    = shape.box(1,1)
    proj.profile  = profile.gaussian(1,.25)
    proj.src      = m0[0]
    proj.dst      = m1[0]
    proj.connect()
    
    
    view = View (m1[0], m0[0], size=8)
    view.show()

