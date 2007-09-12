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
#------------------------------------------------------------------------------
""" OpenGL Python/IPython terminal

    GLPython is basically a Python (or IPython) terminal that is emulated
    inside an OpenGL window. It supports completion, history, ANSI codes as
    well as some GNU readline features. Usual key bindings are available as
    well as scrolling through the mouse scroll button.

    GLPython relies on backends such as GTK, WX, SDL or Qt that are able to
    open an OpenGL context and handle keyboard and mouse events when necessary.
    You are able to choose the backend to be used when you start glpython using
    the '-b' option.

    The size and position of the terminal is pretty flexible: it can takes the
    entire screen or be reduced to a relative or absolute size anywhere on the
    window. Key and mouse events are only processed when mouse is inside the
    terminal area.

    GLPython package also defines an "Object" class that is the base class for
    displaying objects inside the window. You cannot directly issued OpenGL
    command from inside the terminal or you'll certainly get some unexpected
    effects. However, if you define some object with a "render" method and
    append this object to the main viewport ("view"), then it will be
    displayed immediately (hopefully).
"""

import sys
import os.path
import OpenGL.GL as GL
import OpenGL.GL.EXT.framebuffer_object as GL_EXT
import Image as PIL
import numpy
from glpython import backends
from glpython.terminal import Terminal
from glpython.core import *
from glpython.objects import *

window_number = 1


class Figure (Viewport):
    """
    Figure class
    """

    def __init__ (self, **kwargs):
        """
        Create a new empty figure
        
        Function signature
        ------------------
        
        __init__ (...) 

        Keyword arguments
        -----------------        
        
        size  -- Relative or absolute size
        
        position -- Relative or absolute position
        
        has_border -- Whether figure has border or not
        """
        Viewport.__init__(self,**kwargs)

    def set_view (self, theta, phi, zoom):
        """
        Set figure view
        
        Function signature
        ------------------
        
        __init__ (theta, phi, zoom)
        
        Keyword arguments
        ----------------- 
        
        theta -- Rotation around x axis
        
        phi -- Rotation around z axis
        
        zoom -- Zoom level
        """
        
        self.observer.theta = theta
        self.observer.phi = phi
        self.observer.camera.zoom = zoom

    def colorbar (self, **kwargs):
        """
        Add a colorbar to the figure.
        
        Function signature
        ------------------
        
        colorbar (...)
        
        Keyword arguments
        -----------------
        
        cmap -- Colormap to represent
        
        size -- Size along main axis as a percentage of window size
        
        aspect -- Aspect of the colorbar
        
        position -- Relative or absolute position of the colorbar
        
        orientation -- 'vertical' or 'horizontal'
        
        alpha -- Colorbar transparency
        
        name -- Name of the colorbar
        """

        cb = Colorbar (**kwargs)
        self.append (cb)
        return cb

    def background (self, **kwargs):
        """
        Add a background to the figure.
        
        Function signature
        ------------------
        
        background (cmap, ...)
        
        Keyword arguments
        -----------------
        
        orientation -- 'vertical' or 'horizontal'
        
        alpha -- Background transparency
        
        name -- Name of the background
        """

        bg = Background (**kwargs)
        self.append (bg)
        return bg

    def text (self, **kwargs):
        """
        Add a text to the figure.
        
        Function signature
        ------------------
        
        text (...) 
        
        Keyword arguments
        -----------------
        
        text -- Text to be displayed
        
        position -- Relative or absolute position
        
        color -- Color of the text
        
        size -- Relative or absolute size
        
        alignment -- 'left', 'center' or 'right'
        
        orientation -- from 0 to 359 degrees
        
        alpha -- Text transparency
        
        name -- Name of the text
        """

        t = Text (**kwargs)
        self.append (t)
        return t

    def label (self, X, x, y, zscale = .25, elevation=.25, **kwargs):
        """
        Add a label to a surface or an image
        
        Function signature
        ------------------
        
        label (X, x, y, zscale, elevation, ...)
        
        Function arguments
        ------------------
        
        X -- An array of shape MxN
        
        x -- x index within array
        
        y -- y index within array
        
        zscale - zscale of the underlying surface
        
        elevation - how high is placed label above surface
        
        Keyword arguments
        -----------------
        
        text -- Text to be displayed
        
        fg_color -- Color of the text

        bg_color -- Background color
        
        br_color -- Border color
        
        size -- Relative or absolute size
        
        alpha -- Title transparency
        
        name -- Name of the label
        """

        z = float(X[y][x])*.25
        fx, fy = x/float(X.shape[0]-1)-.5, y/float(X.shape[1]-1)-.5        
        if 'text' not in kwargs.keys():
            kwargs['text'] = "(%d,%d) : %.3f" % (x,y,z)
        l = Label (position=(fx,fy,z, z+.25), **kwargs )
        self.append (l)
        return l

    def surface (self, X, style, **kwargs):
        """
        Display a 2 dimensional array as a surface.
        
        Array must be a float32 array, and can have the following shapes:
        MxN    : luminance
        MxNx3  : RGB
        MxNx4  : RGBA

        Function signature
        ------------------
        
        surface (X, style, ...) 

        Function arguments
        ------------------        

        X -- An image filename, a PIL image or an array
        
        style -- 'flat', 'smooth' or 'cubic'
        
        Keyword arguments
        -----------------        
        
        cmap -- Colormap to use for MxN type array
        
        frame -- Normalized frame for display
        
        alpha -- Overall transparency
        """

        if type(X) is str:
            try:
                X = PIL.open (X)
            except:
                print "Cannot load image"
                return
        if isinstance (X, PIL.Image):
            try:
                X = X.transpose (PIL.FLIP_TOP_BOTTOM)
                X = numpy.asarray(X, dtype='float32')
                X = numpy.divide (X, 256.0)
            except:
                print "Cannot convert image to luminance"
                return

        if style in ['smooth', 'cubic']:
            if len(X.shape) != 2:
                print "Array shape must be MxN"
                return
            if style == 'cubic':
                a = CubicSurface (X, **kwargs)
            elif style == 'smooth':
                a = SmoothSurface (X, **kwargs)
        else:
            if ((len(X.shape) != 2 and len(X.shape) != 3) or
                (len(X.shape) == 3 and X.shape[2] != 3 and X.shape[2] != 4)):
                print "Array shape must be either MxN, MxNx3 or MxNx4"
                return
            a = FlatSurface (X, **kwargs)
        self.append (a)
        return a


    def flat_surface (self, X, **kwargs):
        """
        Display a 2 dimensional array as a flat surface.
        
        Array must be a float32 array, and can have the following shapes:
        MxN    : luminance
        MxNx3  : RGB
        MxNx4  : RGBA

        Function signature
        ------------------
        
        flat_surface (X, ...) 

        Function arguments
        ------------------        

        X -- An image filename, a PIL image or an array
        
        Keyword arguments
        -----------------        
        
        cmap -- Colormap to use for MxN type array
        
        frame -- Normalized frame for display
        
        alpha -- Overall transparency
        """
        
        return self.surface (X, 'flat', **kwargs)
        
    def smooth_surface (self, X, **kwargs):
        """
        Display a 2 dimensional array as a smooth surface.
        
        Array must be a float32 array, and can have the following shapes:
        MxN    : luminance

        Function signature
        ------------------
        
        smooth_surface (X, ...) 

        Function arguments
        ------------------        

        X -- An image filename, a PIL image or an array
        
        Keyword arguments
        -----------------        
        
        cmap -- Colormap to use
        
        frame -- Normalized frame for display
        
        alpha -- Overall transparency
        """
        
        return self.surface (X, 'smooth', **kwargs)

    def cubic_surface (self, X, **kwargs):
        """
        Display a 2 dimensional array as a cubic surface.
        
        Array must be a float32 array, and can have the following shapes:
        MxN    : luminance

        Function signature
        ------------------
        
        cubic_surface (X, ...) 

        Function arguments
        ------------------        

        X -- An image filename, a PIL image or an array
        
        Keyword arguments
        -----------------        
        
        cmap -- Colormap to use
        
        frame -- Normalized frame for display
        
        alpha -- Overall transparency
        """
        
        return self.surface (X, 'cubic', **kwargs)


    def figure (self, **kwargs):
        """
        Create a new figure
        
        Function signature
        ------------------
        
        figure (...) 

        Keyword arguments
        -----------------        
        
        size  -- Relative or absolute size
        
        position -- Relative or absolute position
        
        has_border -- Whether figure has border or not
        """
        f = Figure (**kwargs)
        self.append (f)
        return f
        

    def save (self, filename=None, zoom=2):
        """
        Save figure in a file as a bitmap
        
        Function signature
        ------------------
        
        save (filename, zoom) 

        Function arguments
        ------------------        

        filename -- Filename where to save file,
                    format is deduced from filename extension
                    Default: snapshot-xxx.png
        
        zoom -- Zoom to perform relatively to current figure size
                Default: 2
        
        """

        if not filename or os.path.exists(filename):
            i = 0
            filename = "snapshot-%.4d.png" % i
            while os.path.exists (filename):
                i += 1
                filename = "snapshot-%.4d.png" % i

        viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)

        _x,_y,_w,_h = self.geometry
        size = (int(self.geometry[2]*zoom), int(self.geometry[3]*zoom))
        w,h = size[0], size[1]
        size = (w,h)
        image = PIL.new ("RGB", (w, h), (0, 0, 0))
        bits = image.tostring("raw", "RGBX", 0, -1)

        # Setup framebuffer
        framebuffer = GL_EXT.glGenFramebuffersEXT (1)
        GL_EXT.glBindFramebufferEXT (GL_EXT.GL_FRAMEBUFFER_EXT, framebuffer)

        # Setup depthbuffer
        depthbuffer = GL_EXT.glGenRenderbuffersEXT (1)
        GL_EXT.glBindRenderbufferEXT (GL_EXT.GL_RENDERBUFFER_EXT, depthbuffer)
        GL_EXT.glRenderbufferStorageEXT (GL_EXT.GL_RENDERBUFFER_EXT,
                                         GL.GL_DEPTH_COMPONENT, w, h)
        
        # Create texture to render to
        texture = GL.glGenTextures (1)
        GL.glBindTexture (GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MAG_FILTER,
                            GL.GL_LINEAR)
        GL.glTexParameteri (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MIN_FILTER,
                            GL.GL_LINEAR)
        GL.glTexImage2D (GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                         GL.GL_RGB, GL.GL_UNSIGNED_BYTE, bits)
        GL_EXT.glFramebufferTexture2DEXT (GL_EXT.GL_FRAMEBUFFER_EXT,
                                          GL.GL_COLOR_ATTACHMENT0_EXT,
                                          GL.GL_TEXTURE_2D,
                                          texture, 0)
        GL_EXT.glFramebufferRenderbufferEXT (GL_EXT.GL_FRAMEBUFFER_EXT,
                                             GL_EXT.GL_DEPTH_ATTACHMENT_EXT, 
                                             GL_EXT.GL_RENDERBUFFER_EXT,
                                             depthbuffer)
                                    
        status = GL_EXT.glCheckFramebufferStatusEXT (GL_EXT.GL_FRAMEBUFFER_EXT)
        if status != GL_EXT.GL_FRAMEBUFFER_COMPLETE_EXT:
            print "Error in framebuffer activation"
            return

        # Render & save
        GL.glViewport (0, 0, w,h)
        GL.glClearColor (1,1,1,1)
        GL.glClear (GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.resize_event (0,0, w,h)
        GL.glViewport (0, 0, w,h)
        self.render ()
        
        data = GL.glReadPixels (0, 0, w, h, GL.GL_RGB,  GL.GL_UNSIGNED_BYTE)
        image.fromstring (data)
        image = image.crop ((0, 0, w,h))
        image = image.transpose(PIL.FLIP_TOP_BOTTOM)
        image.save (filename)

        # Cleanup
        GL_EXT.glBindRenderbufferEXT (GL_EXT.GL_RENDERBUFFER_EXT, 0)
        GL_EXT.glBindFramebufferEXT (GL_EXT.GL_FRAMEBUFFER_EXT, 0)
        GL.glDeleteTextures (texture)
        GL_EXT.glDeleteFramebuffersEXT (1, [framebuffer])
        GL.glViewport (viewport[0], viewport[1], viewport[2], viewport[3])
        GL.glClear (GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.resize_event (viewport[0], viewport[1], viewport[2], viewport[3])
        print "File has been saved in '%s'" % filename



def window (size=(800,640), fps = 30.0, has_terminal = False, backend = 'wx',
            title = 'Figure', figure = None, namespace = {}, layout = 4):
    """
    Create a new window and return a handle to it as well as the
    underlying default figure.
 
    Function signatures
    -------------------
    
    window (...) ->  (Window, Figure)
    
    Keyword arguments
    -----------------
 
    size -- (w,h) figure size
            default is (800,640) 

    fps -- frames per second (refresh rate)

    has_terminal -- whether to use a terminal inside the window
                    default is false

    backend -- backend to use if possible ('wx', 'gtk' or 'sdl')
    
    title -- window title
    
    fig -- Figure to use (may be None)
    
    namespace -- namespace to use for the terminal
    
    layout -- Window layout to use
    """

    global window_number
    base,backend = backends.WindowClass ('wx')

    class Window (base):
        """
        """

        def __init__ (self, w=800, h=640, title=None, figure=None):
            """
            """
            base.__init__ (self,w,h,title)
            self.initialized = False
            self.outer_mainloop = False
            if not isinstance (figure, Figure):
                self.figure = Figure ()
            else:
                self.figure = figure
            self.figure.has_border = False
            self.terminal = None
            self.layout = 1

        def init (self):
            """ """
            GL.glClearColor (1.0, 1.0, 1.0, 1.0)
            GL.glShadeModel (GL.GL_SMOOTH)
            GL.glEnable (GL.GL_DEPTH_TEST)
            GL.glEnable (GL.GL_NORMALIZE)
            self.figure.init()
            if self.terminal:
                self.terminal.init()

        def resize (self, w, h):
            if not self.initialized:
                self.init()
                self.initialized = True
            self.width, self.height = w,h
            GL.glViewport (0, 0, w, h)
            self.figure.resize_event (0,0,w,h)
            if self.terminal:
                self.terminal.resize (w,h)

        def render (self):
            GL.glClearColor (1.0, 1.0, 1.0, 1.0)
            GL.glClear (GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
            if self.layout in [1]:
                if self.terminal:
                    self.terminal.render()
                self.figure.render ()
            else:
                self.figure.render ()
                if self.terminal:
                    self.terminal.render()
            

        def key_press (self, key):
            """ """
            if key == 'control-d':
                if self.terminal and self.terminal.rl.line == '':
                    self.destroy()
            elif key == 'control-f' or key == 'f11':
                if self.fullscreened:
                    self.unfullscreen()
                    self.fullscreened = False
                else:
                    self.fullscreen()
                    self.fullscreened = True
            elif key in ['f1', 'f2', 'f3', 'f4'] and self.terminal:
                layouts = {'f1': 1, 'f2':2, 'f3':3, 'f4':4}
                self.set_layout (layouts[key])
            elif self.terminal:
                self.terminal.key_press (key)

        def button_press (self, button, x, y):
            if button == 3:
                self.figure.button_press_event (button,int(x),int(y))
                self.figure.select_event (int(x),int(y))
            elif button == 4 and self.terminal:
                self.terminal.scroll_up()
            elif button == 5 and self.terminal:
                self.terminal.scroll_down()            
            else:
                self.figure.button_press_event (button,int(x),int(y))

        def button_release (self, x, y):
            self.figure.button_release_event (0,int(x),int(y))

        def mouse_motion (self, x, y):
            if self.terminal:
                self.terminal.check_focus (x,y)
            self.figure.focus_event(x,y)
            self.figure.pointer_motion_event (x,y)
            if self.figure.has_focus and self.terminal:
                self.terminal.focus = False

        def set_layout (self, layout):
            if not self.terminal:
                return

            c = self.terminal.bg_color
            ct = (c[0], c[1], c[2], .75) # transparent
            co = (c[0], c[1], c[2], 1)   # opaque

            # Terminal fullscreen, behind
            if layout == 1:
                self.figure.size = (1.0,1.0)
                self.figure.position = (0,0)
                self.figure.has_border = False                
                self.terminal.size = (1.0,1.0)
                self.terminal.position = (0,0)
                self.terminal.has_border = False
                self.terminal.set_bg_color (co)
                self.layout = 1
                
            # Terminal fullscreen, front
            elif layout == 2:
                self.figure.size = (1.0,1.0)
                self.figure.position = (0,0)
                self.figure.has_border = False                
                self.terminal.size = (1.0,1.0)
                self.terminal.position = (0,0)
                self.terminal.has_border = False
                self.terminal.set_bg_color (ct)
                self.layout = 2

            # Terminal bottom, figure above
            elif layout == 3:
                self.terminal.size = (1.0,0.30)
                self.terminal.position = (0,0)
                self.terminal.has_border = True
                self.terminal.set_bg_color (co)
                self.figure.size = (1.0,.70)
                self.figure.position = (0,-1)
                self.layout = 3

            # Terminal bottom, figure fullscreen
            else:
                self.terminal.size = (1.0,0.30)
                self.terminal.position = (0,0)
                self.terminal.has_border = True
                self.terminal.set_bg_color (ct)
                self.figure.size = (1.0,1.0)
                self.figure.position = (0,0)
                self.layout = 4

            if self.initialized:
                self.resize (self.width, self.height)

    w = size[0]
    h = size[1]
    if not title:
        title = "Figure %d" % window_number
    window_number += 1
    win = Window (w, h, title, figure)
    if has_terminal:
        namespace['window'] = win
        namespace['figure'] = win.figure
        win.terminal = Terminal (namespace, 'ipython', win.paint, win.idle)
        win.terminal.has_border = False
        win.terminal.shell.namespace['terminal'] = win.terminal
    win.set_layout (layout)
    return win, win.figure

