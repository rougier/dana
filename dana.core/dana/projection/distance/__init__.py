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
# $Id: __init__.py 119 2007-02-07 14:16:22Z rougier $
#------------------------------------------------------------------------------
""" Distance functions

Distance functions implements various distances that can be used for measuring
the distance between a source and a target.

"""

from _distance import *
__all__ = ['distance', 'euclidean', 'manhattan', 'max']
