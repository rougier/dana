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
""" GTK graphical user interface """

import gtk
from control_panel import *
from viewer import *

__all__ = ['ControlPanel', 'Edit', 'View', 'Viewer']

gtk.gdk.threads_init()
