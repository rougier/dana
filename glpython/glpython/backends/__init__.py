#!/usr/bin/env python

#------------------------------------------------------------------------------
#
#   Copyright (c) 2007 Nicolas Rougier
# 
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#------------------------------------------------------------------------------

""" OpenGL backends

This module provides a Window class that create a graphical window with an
OpenGL context. The created window has several functions that can be used to
control how a scene is displayed and how to interact. Those functions are
independent of the used backend and should be overloaded by the user.

The backend is chosen automatically among those available.
"""


backends = []
__all__ = ['Window']

try:
    import pygame, backend_sdl
    backends.append ('sdl')
except:
    pass

try:
    import gtk, backend_gtk
    backends.append ('gtk')
except:
    pass

try:
    import wx, backend_wx
    backends.append ('wx')
except:
    pass

_backend = None

class BackendError(Exception):
    """ Base class for exceptions in this module """

    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

if not backends:
    raise BackendError ("No backend available")
elif 'wx' in backends and wx.App.IsMainLoopRunning():
    print 'A WX event loop is running, using WX backend'
    outer_mainloop = True
    _backend = 'WX'
elif 'gtk' in backends and gtk.main_level():
    print 'A GTK event loop is running, using GTK backend'
    outer_mainloop = True
    _backend = 'GTK'


def Window (w=800, h=600, title='OpenGL window', fps=30.0, backend = 'wx'):
    """ Create window with a specific backend """
    
    if _backend:
        backend = _backend
    if backend not in backends:
        print '%s backend not available, using %s' % (backend, backends[-1])
        backend = backends[-1]
    if backend == 'sdl':
        return backend_sdl.Window (w,h,title +' (%s backend)' % backend,fps)
    elif backend == 'gtk':
        return backend_gtk.Window (w,h,title +' (%s backend)' % backend,fps)
    return backend_wx.Window (w,h,title +' (%s backend)' % backend,fps)


def WindowClass (backend = 'wx'):
    """ Return the proper window class according to backend and those
        available """

    if _backend:
        backend = _backend
    if backend not in backends:
        print '%s backend not available, using %s' % (backend, backends[-1])
        backend = backends[-1]    
    if backend == 'sdl':
        return backend_sdl.Window, backend
    elif backend == 'gtk':
        return backend_gtk.Window, backend
    return backend_wx.Window, backend

