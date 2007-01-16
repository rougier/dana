#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006, Nicolas Rougier.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under the
# conditions described in the aforementioned license. The license is also
# available online at http://www.loria.fr/~rougier/pub/Licenses/BSD.txt
# 
# $Id$
#------------------------------------------------------------------------------

""" Projection tools

A projection is the specification of a pattern of connection between three layers.
It can be precisely defined using :
  - a combination function : it defines how to combine the inputs to define
			     the links to each destination's neuron
"""

from dana.sigmapi._sigmapi import *
from dana.sigmapi.projection._projection import *