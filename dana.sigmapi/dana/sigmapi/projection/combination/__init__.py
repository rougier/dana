#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006, Nicolas Rougier
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under the
# conditions described in the aforementioned license. The license is also
# available online at http://www.loria.fr/~rougier/pub/Licenses/BSD.txt
# 
#  $Id$
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

#from dana.sigmapi._sigmapi import *
#from dana.sigmapi.projection._projection import *
from _combination import *

__all__ = ['Combination','Linear']
