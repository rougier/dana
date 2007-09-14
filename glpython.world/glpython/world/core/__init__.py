#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier - Jeremy Fix
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------
""" GLPython world

"""
from glpython.core import *
from _core import *

import sys
import os.path
import OpenGL.GL as GL
import OpenGL.GL.EXT.framebuffer_object as GL_EXT
import Image as PIL


__all__ = ['Viewport','Robot','Observer','Camera']


def save (self,filename,save_width,save_height):

    viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)

    _x,_y,_w,_h = self.geometry

    print "Size avant resize %f, %f \n" % (_w,_h)

    size = (save_width,save_height)

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
    print "Size %f, %f \n" % (self.geometry[2],self.geometry[3])
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
    #print "File has been saved in '%s'" % filename


# _________________________________________________________________________Robot
# The Robot class is overloaded because the save method of the viewport is dynamically modified when python is executed

class Robot (_core.Robot):
    """
    Robot class

    Extends the C++ class Robot to provide it with the grab method
    """

    def __init__ (self, name="roger"):
        """

        Create a new robot
        
        """

        _core.Robot.__init__ (self,name)

    def grab(self,filename,grab_width,grab_height):
        self.view.save(filename,grab_width,grab_height)



_core.Viewport.save = save

