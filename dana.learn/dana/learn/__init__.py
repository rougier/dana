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
""" Learning package
This package implements learning rules derived from the rule dwij/dt = F(wij,vi,vj)
A learning rule is defined as a sum of elementary blocks.
An elementary block of the learning rule is defined with powi, powj and an array representing a polynomial
function of the weight : P(w).(vi**powi).(vj**powj)
The array defining P(w) is constructed as following : [polynomial function of w] = [a0,a1,a2,...]\
with : P(w) = sum_i (ai.w**i)

This package implements learning rules derived from the rule dwij/dt = F(wij,vi,vj)
"""

from dana.core import *
from dana.cnft import *
from _learn import *

__all__ = ['Unit','MUnit','SUnit']
