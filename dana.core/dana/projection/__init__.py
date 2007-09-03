#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: __init__.py 119 2007-02-07 14:16:22Z rougier $
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

from distance import *
from profile import *
from shape import *
from density import *
from _projection import *


def one_to_one (src, dst, w=1.0, self_connect=False):
    """ One to one connection """

    return Projection (src, dst,
                       shape       = Point(),
                       distance    = Euclidean(),
                       density     = Full(),
                       profile     = Constant(w),
                       self_connect= self_connect)

def linear (src, dst, w=1.0, self_connect=False):
    """ Linear connection """

    return Projection (src, dst,
                       shape       = Box (1.0, 0.0),
                       distance    = Euclidean(),
                       density     = Full(),
                       profile     = Constant(w),
                       self_connect= self_connect)

def columnar (src, dst, w=1.0, self_connect=False):
    """ Columnar connection """

    return Projection (src, dst,
                       shape       = Box (0.0, 1.0),
                       distance    = Euclidean(),
                       density     = Full(),
                       profile     = Constant(w),
                       self_connect= self_connect)

def all_to_one (src, dst, w=1.0, self_connect=False):
    """ All to one connection """

    return Projection (src, dst,
                       shape       = Box (1.0, 1.0),
                       distance    = Euclidean(),
                       density     = Full(),
                       profile     = Constant(w),
                       self_connect= self_connect)

def gaussian (src, dst, a=1.0, b=1.0, self_connect=False):
    """ Gaussian connection """

    return Projection (src, dst,
                       shape       = Box (1.0, 1.0),
                       distance    = Euclidean(),
                       density     = Full(),
                       profile     = Gaussian(a,b),
                       self_connect= self_connect)

def dog (src, dst, a1=1.0, b1=1.0, a2=.5, b2=.5, self_connect=False):
    """ Difference of Gaussian connection """

    return Projection (src, dst,
                       shape       = Box (1.0, 1.0),
                       distance    = Euclidean(),
                       density     = Full(),
                       profile     = DoG(a1,b1,a2,b2),
                       self_connect= self_connect)
