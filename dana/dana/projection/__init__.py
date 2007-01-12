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
""" projection tools

A projection is the specification of a pattern of connection between two layers.
It can be precisely defined using four different notions:

  - a distance : it defines how to measure distances between a source and a
                 target and can be either the euclidean, the manhattan or the
                 max distance. Each distance can be made toric.
                 
  - a shape    : it defines the most general set of sources that can
                 potentially be connected to a target. It can be a point, 
                 a box of a given size or a disc of a given radius.
  - a profile  : it defines connection weights as a function of the distance
                 between a source and a target.
  - a density  : it defines the probability of a connection to be actually
                 instantiated as a function of the distance.
"""

from _projection import *
import density
import distance
import profile
import shape

__all__ = ['projector', 'projection', 'density', 'distance', 'profile', 'shape']
