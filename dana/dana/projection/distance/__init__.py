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
# Author: Nicolas Rougier
# Description: DANA library component
#------------------------------------------------------------------------------
""" Distributed Asyncronous Numerical Adaptive Computing Library

Distributed Asyncronous Numerical Adaptive Computing Library
============================================================

For a complete definition of what is DANA computing, please have a look at
the article "DANA Computing - Foundations".

"""

from _distance import *
__all__ = ['distance', 'euclidean', 'manhattan', 'max']
