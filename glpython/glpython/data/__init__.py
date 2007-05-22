#!/usr/bin/env python

#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: __init__.py 146 2007-05-10 14:19:21Z rougier $
#------------------------------------------------------------------------------

import os.path
def datadir():
    mod_loc = os.path.dirname(__file__)
    return os.path.abspath(mod_loc)
