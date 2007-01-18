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
""" Profile functions

A profile function describes the weight of a link between a source and a target
as a function of their distance.

"""

from _profile import *
__all__ = ['profile', 'constant', 'linear', 'gaussian', 'dog', 'uniform']
