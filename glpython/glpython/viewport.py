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

""" Viewport object

A viewport represents a rectangular sub-area of the main GL viewport. It is
meant to be an 'independent' view and can dispatch received events.

"""

from OpenGL.GL import *
from OpenGL.GL.EXT.framebuffer_object import *
import Image
from observer import Observer


class Viewport(list):
    """ """

    def __init__(self, position=(0,0), size=(1.0,1.0),
                 name='Viewport', reset=None):
        """ Initializes the frame """
        
        list.__init__(self)
        
        self.visible = True
        self.active = True
        self.focus = False
        self.depth = 1
        self.name = name
        self.button = 0
        self.reset = reset
        
        self.position = position
        self.size = size
        self.geometry = (0,0,10,10)

        self.use_border = True
        self.bg_color = [1,1,1,.75]
        self.br_color = [0,0,0,1]

        self.observer = Observer()


    def set_position (self, x, y):
        """ Set position """

        self.position = (x,y)
        x,y,w,h = self.geometry
        if self.reset:
            self.reset()

    def set_size (self, w, h):
        """ Set size """

        self.size = (w,h)
        x,y,w,h = self.geometry
        if self.reset:
            self.reset()

    def hide (self):
        """ Hide viewport """

        self.visible = False

    def show (self):
        """ Show viewport """

        self.visible = True

    def selection (self,x,y):
        """ pre-selection event """
        
        self.observer.camera.x = x
        self.observer.camera.y = y
        for o in self:
            if hasattr(o,'selection'):
                o.selection (x,y)

    def select (self, id1, id2):
        """ Select event """

        for o in self:
            o.select (id1, id2)
            
    def unselect (self):
        """ Unselect event """

        for o in self:
            o.unselect ()

    def init (self):
        """ Initialization event """

        glClearColor (1,1,1,1)
        glShadeModel (GL_SMOOTH)
        glEnable (GL_DEPTH_TEST)
        glEnable (GL_NORMALIZE)
        glMaterial (GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterial (GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glMaterial (GL_FRONT, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glLight (GL_LIGHT0, GL_AMBIENT, [0.25, 0.25, 0.25, 1.0])
        glLight (GL_LIGHT0, GL_DIFFUSE, [0.75, 0.75, 0.75, 1.0])
        glLight (GL_LIGHT0, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glEnable (GL_LIGHTING)
        glEnable (GL_LIGHT0)
        glLight (GL_LIGHT0, GL_POSITION, [4, 1.0, 4.0, 0.0])
        glMaterialfv (GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [1,1,1])
        glColor ([0,0,0])        
        for o in self:
            o.init()


    def has_focus (self):
        """ Return focus status """
        
        focus = self.focus
        for o in self:
            if hasattr(o,'has_focus') and o.has_focus() and o.visible:
                focus = True
        return focus


    def check_focus (self, x, y):
        """ Check if viewport has focus """
        
        if self.button:
            self.focus = True
            return
        
        xx,yy,w,h = self.geometry
      	viewport = glGetIntegerv (GL_VIEWPORT)
        xx += self.ox
        yy += self.oy

        if x > xx and x < (xx+w) and y > yy and y < (yy+h):
            self.focus = True
        else:
            self.focus = False
        focus = self.focus
        for o in self:
            if hasattr(o,'check_focus') and o.check_focus(x,y) and o.visible:
                self.focus = False
                focus = True
        return focus


    def key_press (self, key):
        """ Key press event """

        if not self.visible:
            return

        if self.focus:
            if key == 'f10':
                self.screenshot ("screenshot.png", 2)
            elif key == 'f9':
                self.observer.camera.ortho = not self.observer.camera.ortho
            elif key == 'escape':
                self.observer.camera.ortho = True
                self.observer.theta = 0
                self.observer.phi = 0
                self.observer.camera.zoom = 1.0
        else:
            for o in self:
                o.key_press(key)


    def button_press (self, button, x,y):
        """ Mouse button press event """

        if not self.visible:
            return

        self.check_focus(x,y)
        
        if self.focus:
            self.button = button
            if button == 1 or button == 2:
                self.observer.move_start (x, y, button)
        else:
            for o in self:
                o.button_press (button,x,y)


    def button_release (self, x,y):
        """ Mouse button release event """

        if not self.visible:
            return

        if self.button:
            self.observer.move_end (x, y)
            self.button = 0
        
        for o in self:
            o.button_release (x,y)
        self.check_focus(x,y)


    def mouse_motion (self, x, y):
        """ Mouse motion event """

        if not self.visible:
            return

        if self.focus and self.button:
            self.observer.move_to (x,y)
        else:
            for o in self:
                o.mouse_motion (x,y)
            self.check_focus(x,y)


    def selection (self,x,y):
        """ """
        
        self.observer.camera.x = x
        self.observer.camera.y = y
        for o in self:
            if hasattr(o,'selection'):
                o.selection (x,y)


    def resize (self, w, h):
        """ Check viewport size """

        def round2 (n):
            """ Get nearest power of two superior to n """
            f = 1
            while f<n:
                f*= 2
            return f
 
        viewport = glGetIntegerv (GL_VIEWPORT)
        self.ox, self.oy = viewport[0], viewport[1]

        # Check if actual size match specifications
        width = height = 0
        if type(self.size[0]) is int:
            if self.size[0] < 0:
                width = w + self.size[0]
            else:
                width = self.size[0]
        else:
            if self.size[0] >= 0 and self.size[0] < 1:
                width = int(w*self.size[0])
            else:
                width = w

        if type(self.size[1]) is int:
            if self.size[1] < 0:
                height = h + self.size[1]
            else:
                height = self.size[1]
        else:
            if self.size[1] >= 0 and self.size[1] < 1:
                height = int(h*self.size[1])
            else:
                height = h

        if width < 5:
            width = 5
        if height < 5:
            height = 5

        # Computes position
        x = y = 0
        if type(self.position[0]) is int:
            if self.position[0] < 0:
                x = w + self.position[0] - width
            else:
                x = self.position[0]
        else:
            if self.position[0] >= 0 and self.position[0] <= 1:
                x = int(w*self.position[0])
            elif self.position[0] >= -1 and self.position[0] <= 0:
                x = w + int(w*self.position[0]) - width

        if type(self.position[1]) is int:
            if self.position[1] < 0:
                y = h + self.position[1] - height
            else:
                y = self.position[1]
        else:
            if self.position[1] >= 0 and self.position[1] <= 1:
                y = int(h*self.position[1])
            elif self.position[1] >= -1 and self.position[1] <= 0:
                y = h + int(h*self.position[1]) - height
        self.geometry = (x,y,width,height)

        for o in self:
            glViewport (x, y, width, height)
            o.resize (width,height)


    def render (self):
        """ Setup viewport for rendering """
        
        if not self.visible:
            return

        glPushAttrib (GL_ENABLE_BIT | GL_VIEWPORT_BIT)

      	viewport = glGetIntegerv (GL_VIEWPORT)
        w = viewport[2] - viewport[0]
        h = viewport[3] - viewport[1]

        x,y,w,h = self.geometry       
        glViewport (viewport[0]+x,viewport[1]+y,w,h)
        glEnable (GL_SCISSOR_TEST)
        glScissor (viewport[0]+x,viewport[1]+y,w,h)
        glClear (GL_DEPTH_BUFFER_BIT)
        glDisable (GL_SCISSOR_TEST)

        self.observer.resize (w,h)
        self.observer.look()
        for o in self:
            o.render()

        glPopAttrib()


    def screenshot (self, filename, zoom=1):
        """ Take a screenshot """

        def round2 (n):
            """ Get nearest power of two superior to n """
            f = 1
            while f<n:
                f*= 2
            return f

        X,Y,W,H = self.geometry
        viewport = glGetIntegerv (GL_VIEWPORT)

        size = (int(W*zoom), int(H*zoom))
        position = (int(X*zoom), int(Y*zoom))
        w = round2 (size[0])
        h = round2 (size[1])

        image = Image.new ("RGB", (w, h), (0, 0, 0))
        bits = image.tostring("raw", "RGBX", 0, -1)

        # Setup framebuffer
        framebuffer = glGenFramebuffersEXT (1)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer)

        # Setup depthbuffer
        depthbuffer = glGenRenderbuffersEXT (1)
        glBindRenderbufferEXT (GL_RENDERBUFFER_EXT,depthbuffer)
        glRenderbufferStorageEXT (GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, w, h)
        
        # Create texture to render to
        texture = glGenTextures (1)
        glBindTexture (GL_TEXTURE_2D, texture)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                        GL_RGB, GL_UNSIGNED_BYTE, bits)
        glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                                   GL_TEXTURE_2D, texture, 0);
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
                                     GL_RENDERBUFFER_EXT, depthbuffer);
                                    
        status = glCheckFramebufferStatusEXT (GL_FRAMEBUFFER_EXT);
        if status != GL_FRAMEBUFFER_COMPLETE_EXT:
            print "Error in framebuffer activation"
            return

        # Render & save
        _position, _size, _geometry = self.position, self.size, self.geometry
        self.position = (0,0)
        self.size = (1.0,1.0)
        
        glViewport (0, 0, size[0],size[1])
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.resize (size[0], size[1])
        glViewport (0, 0, size[0], size[1])
        self.render ()

        self.position, self.size, self.geometry = _position, _size, _geometry

        data = glReadPixels (0, 0, w, h, GL_RGB,  GL_UNSIGNED_BYTE)
        image.fromstring (data)
        
        image = image.crop ((0, 0, size[0], size[1]))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        try:
            image.save (filename)
        except:
            print 'Cannot save screenshot file'

        # Cleanup
        glBindRenderbufferEXT (GL_RENDERBUFFER_EXT, 0)
        glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0)
        glDeleteTextures (texture)
        glDeleteFramebuffersEXT (1, [framebuffer])

        if self.reset:
            self.reset()



