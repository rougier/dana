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

import dana.core as core
import dana.projection as proj

size = 5;

net = core.Network()
    
m0 = core.Map( (size, size), (0,0) )
m0.append( core.Layer() )
m0[0].fill(core.Unit)
net.append(m0)

m1 = core.Map( (size, size), (1,0) )
m1.append( core.Layer() )
m1[0].fill(core.Unit)
net.append(m1)


p          = proj.projection()
p.self     = True
p.distance = proj.distance.euclidean(False)
p.density  = proj.density.full(1)
p.shape    = proj.shape.point()
p.profile  = proj.profile.constant(1)
p.src      = m0[0]
p.dst      = m1[0]
p.connect()

net.write("net.gz")
net.read("net.gz")

