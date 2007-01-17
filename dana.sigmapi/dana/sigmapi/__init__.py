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

""" Sigma Pi Neurons

The SigmaPi library implements Sigma-Pi type Neurons
"""

from dana.core import *
from _sigmapi import *

__all__ = ['Link','Unit']
