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
""" Density functions

A density is a function of the distanc between a source and a target describing
the probability of the link betwwen source and target to be made.

"""

from _density import *
__all__ = ['density', 'full', 'sparse', 'sparser']
