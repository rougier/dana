#!/usr/bin/env python

#------------------------------------------------------------------------------
#
#   Copyright (c) 2007 Nicolas P. Rougier
# 
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
# 
#------------------------------------------------------------------------------

""" Interactive OpenGL python terminal

This implements an interactive python session in an OpenGL window by first
rendering it into a texturte using framebuffer. Terminal is supposed to
receive key event as well as resize event to function properly.

Shortcuts:
----------
    Ctrl-A : goto line start
    Ctrl-E : goto line end
    Ctrl-K : clear line from cursor to end
    Ctrl-L : clear console
    Ctrl-S : save session
    Tab:     completion
"""

from terminal import Terminal
__all__ = ['Terminal']
