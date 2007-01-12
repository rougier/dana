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
""" random library

Distributed Asyncronous Numerical Adaptive Computing Library
============================================================

The random module provides some random generators based on the boost random
library.

"""
from _random import *

all = ['uniform', 'normal', 'seed']
