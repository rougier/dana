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
# $Id: __init__.py 166 2007-05-18 07:54:33Z rougier $
#------------------------------------------------------------------------------
""" core component

From a conceptual point of view, the computational paradigm supporting the
library is grounded on the notion of a unit that is essentially a potential
that can vary along time under the influence of other units and learning.
Those units are organized into layers, maps and network: a network is made
of one to several map, a map is made of one to several layer and a layer is
made of a set of several units. Each unit can be linked to any other unit
(included itself) using a weighted link.

  Unit
  Link
  Layer
  Map
  Network
  Spec
  Environment
  Model
  

"""

import sys, os.path
from _core import *
import random

_random = Random()

def seed (s = None):
    """ Seed for internal random generator """

    if not s:
        random.seed (_random.seed)
        return _random.seed
    random.seed (s)
    _random.seed = s
    return _random.seed

seed (12345)


Object.__write = Object.write
def __write (self, filename):
    """ """
    co = sys._getframe(1).f_code
    script = "\n"
    f = file (co.co_filename, "r")
    for line in f:
        script += line
    f.close()
    if filename[-3:] != '.gz':
        filename += '.gz'
    self.__write (filename,  os.path.abspath(co.co_filename), script)
Object.write = __write

