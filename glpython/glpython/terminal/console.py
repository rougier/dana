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

""" Console View

This console implements a console object using OpenGL. The console is
actually rendered to a texture using the framebuffer and is then
displayed onto a regular quad using ortho mode. It does not handle key
nor mouse events and it is actually only possible to write lines and move
cursor.
"""

from OpenGL.GL import *
from OpenGL.GL.EXT.framebuffer_object import *
from sterminal import Terminal


class Console:
    """ """

    def __init__(self):
        """ Initializes the console """
        
        self.terminal = Terminal ()
        self.position = (0,0)
        self.size = (1.0,1.0)
        self.bg_color = (1,1,1,1)
        self.fg_color = (0,0,0)
        self.border = 3
        self.default_font_size = 12
        self.visible = True
        self.active = True
        self.depth = 99
        self.focus = False
        self.logo = False

        # Internal
        self.scroll = 0
        self.font = None
        self.lines = 0
        self.columns = 0
        self.width = 0
        self.height = 0
        self.framebuffer = 0
        self.tex_id = 0
        self.tex_w = 0
        self.tex_h = 0
        self.tex_st = (0,0)
        self.geometry = 0,0,5,5

    def set_position (self, x, y):
        """ Set console position
        
        Console position can be specified in many different ways depending
        on the type of and sign of parameters. If a parameter is an integer
        then the position is considered to be in pixels and the sign
        indicates which side of the console box to consider. For x, a
        positive sign means a left border attachement while a negative sign
        means a right border attachement. For y, a positive sign means a
        bottom border attachement while a negative sign means a top
        border attachement. If the type of the parameter is real and
        absolute value is between 0 and 1, then the position is considered
        to be a percentage of window size. The sign has the same
        signification as for integer positions.
        """

        self.position = (x,y)
        self.dirty = True


    def set_size (self, w, h):
        """ Set console size
        
        Console size can be specified in many different ways depending on
        the type of and sign of parameters. If a parameter is an integer
        then the size is considered to be in pixels and the sign indicates
        either an absolute size or a complementary size relative to window
        size. If the type of the parameter is real and absolute value is
        between 0 and 1 then the size if considered to be a percentage of
        window size.
        """

        self.size = (w,h)
        self.dirty = True


    def set_char_size (self, w, h):
        """ Set character console size
        
        Console character size is specified terms of lines and colums.
        """
        
        if not self.font:
            return
        self.size = (w*self.font.size[0]+2*self.border, 
                     h*self.font.size[1]+2*self.border)
        self.reshape()
        self.dirty = True


    def set_bg_color (self, color):
        """ Set console background color
        
        Console color must be specified as a (red,green,blue,alpha) tuple
        of float between 0 and 1. An alpha value of 0 means a fully
        transparent background while a value of 1 means a fully opaque
        background.
        """
        
        self.bg_color = color
        self.dirty = True


    def set_fg_color (self, color):
        """ Set console foreground (text) color
        
        Console color must be specified as a (red,green,blue) tuple of
        float between 0 and 1.
        """
        
        self.fg_color = color
        if self.font:
            self.font._make_color_marker (ord(Black), color)
        self.dirty = True


    def set_font_size (self, size):
        """ Set console font size
        
        Font size must be a non null integer
        """
        
        if size == 0:
            size = self.default_font_size
        self.font = Font (size)
        self.reshape (force=True)


    def scroll_down (self):
        """ Scroll console down """
        
        self.scroll = max (self.scroll-1, 0)
        self.dirty = True


    def scroll_up (self):
        """ Scroll console up """
        
        self.scroll += 1
        self.dirty = True


    def init (self):
        """ Initialize console
        
        Initialize function must be called as soon as an OpenGL context is
        available.
        """
        
        self.font = Font (self.default_font_size)
        self.set_fg_color (self.fg_color)
        self.dirty = True
        
        im = Image.open (os.path.join(datadir(),'glpython.png'))
        ix, iy, image = im.size[0], im.size[1], im.tostring("raw", "RGBA", 0, -1)
        self.logo_texid = glGenTextures(1)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.logo_texid)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, 4, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)


    def check_focus (self, x, y):
        """ Check if console has focus """
        
        xx,yy,w,h = self.geometry
        if x > xx and x < (xx+w) and y > yy and y < (yy+h):
            self.focus = True
        else:
            self.focus = False


    def reshape (self, w=None, h=None, force=False):
        """ Check console dynamic size
        
        Since console size may be specified relative to window size, we
        have to calculate size each time a window resize event is issued
        and allocate a new texture if necessary (since texture size is a
        power of two, it may be sufficient to contain the new requested
        size).
        """
        
        def round2 (n):
            """ Get nearest power of two superior to n """
            f = 1
            while f<n:
                f*= 2
            return f

        if not w or not h:
      	    viewport = glGetIntegerv (GL_VIEWPORT)
            w = viewport[2] - viewport[0]
            h = viewport[3] - viewport[1]

        max_w = max_h = glGetIntegerv (GL_MAX_TEXTURE_SIZE)
        
        # Check if actual console size match specifications
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
                width = w-1

        if type(self.size[1]) is int:
            if self.size[1] < 0:
                height = h + self.size[1]
            else:
                height = self.size[1]
        else:
            if self.size[1] >= 0 and self.size[1] < 1:
                height = int(h*self.size[1])
            else:
                height = h-1

        # Check if we're below card specs
        if width > max_w:
            width = max_w
        if height > max_h:
            height = max_h

        # No change needed
        if self.width == width and self.height == height and not force:
            return
        
        # Update actual size
        self.width = width
        self.height = height

        # Computes console position
        x = y = 0
        if type(self.position[0]) is int:
            if self.position[0] < 0:
                x = w + self.position[0] - self.width
            else:
                x = self.position[0]
        else:
            if self.position[0] >= 0 and self.position[0] <= 1:
                x = int(w*self.position[0])
            elif self.position[0] >= -1 and self.position[0] <= 0:
                x = w + int(w*self.position[0]) - self.width

        if type(self.position[1]) is int:
            if self.position[1] < 0:
                y = h + self.position[1] - self.height
            else:
                y = self.position[1]
        else:
            if self.position[1] >= 0 and self.position[1] <= 1:
                y = int(h*self.position[1])
            elif self.position[1] >= -1 and self.position[1] <= 0:
                y = h + int(h*self.position[1]) - self.height
        self.x, self.y = x,y

        # Compute console size in characters
        self.lines   = (self.height-2*self.border)/self.font.size[1]
        self.columns = (self.width-2*self.border)/self.font.size[0]
        self.terminal.lines = self.lines
        self.terminal.columns = self.columns

        # If framebuffer is already bound, we keep the old texture
        # if glGetIntegerv (GL_FRAMEBUFFER_BINDING_EXT):
        #     return
        if not self.active:
            return
        
        # We use power of two textures so maybe current texture is big enough
        #  for current size
        if self.width <= self.tex_w and self.height <= self.tex_h:
            self.tex_st = [self.width/float(self.tex_w),
                           self.height/float(self.tex_h)]
            self.refresh()
            return

        # No chance, we've to create a whole new texture        
        self.tex_w = round2 (self.width)
        self.tex_h = round2 (self.height)
        self.tex_st = [self.width/float(self.tex_w),
                       self.height/float(self.tex_h)]
        if self.tex_id:
            glDeleteTextures (self.tex_id)
            glDeleteFramebuffersEXT (1, [self.framebuffer])
        image = Image.new ("RGB", (self.tex_w, self.tex_h), (0, 0, 0))
        bits = image.tostring("raw", "RGBX", 0, -1)

        self.framebuffer = glGenFramebuffersEXT (1) #[0]
        
        self.tex_id = glGenTextures (1)
        glBindTexture (GL_TEXTURE_2D, self.tex_id)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, self.tex_w, self.tex_h, 0,
                      GL_RGB, GL_UNSIGNED_BYTE, bits)
        image = None
        bits = None
        self.dirty = True


    def render (self):
        """ Render console
        
        Console render is quite easy, it's basically a quad in ortho view 
        with the console surface textured onto it.
        """

        if self.dirty:
            self.refresh()
            self.dirty = False

      	viewport = glGetIntegerv (GL_VIEWPORT)
        w = viewport[2] - viewport[0]
        h = viewport[3] - viewport[1]
        
        # "2D" projection mode
        glMatrixMode (GL_PROJECTION)
        glPushMatrix ()
        glLoadIdentity ()
        glOrtho (0, w, 0, h, -100, 100)
        glMatrixMode (GL_MODELVIEW)
        glPushMatrix ()
        glLoadIdentity ()
        
        glPushAttrib (GL_ENABLE_BIT | GL_VIEWPORT_BIT)
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL)
        glDisable (GL_LIGHTING)
        glDisable (GL_LINE_SMOOTH)
        glDisable (GL_CULL_FACE)
        glEnable (GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        
        # Computes console position
        x, y = self.x, self.y
        w,h = self.width, self.height
        s,t = self.tex_st

        glEnable (GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glColor4f (1, 1, 1, 1)

        # Console
        glTranslatef (0.0, 0.0, self.depth)
        glBegin (GL_QUADS)
        glTexCoord2f(0, 0), glVertex2f (x, y)
        glTexCoord2f(s, 0), glVertex2f (x+w, y)
        glTexCoord2f(s, t), glVertex2f (x+w, y+h)
        glTexCoord2f(0, t), glVertex2f (x, y+h)
        glEnd()
        
        # Logo 
        if self.logo:
            b = 16
            glBindTexture(GL_TEXTURE_2D, self.logo_texid)
            glColor4f (1, 1, 1, .5)
            glTranslatef (0.0,0.0,0.1)            
            glEnable(GL_SCISSOR_TEST)
            glScissor (x,y,w,h)
            glBegin (GL_QUADS)
            glTexCoord2f (0.0,0.01), glVertex2f (x+w-128-b, b)
            glTexCoord2f (1.0,0.01), glVertex2f (x+w-b, b)
            glTexCoord2f (1.0,1.0), glVertex2f (x+w-b, 128+b)
            glTexCoord2f (0.0,1.0), glVertex2f (x+w-b-128, 128+b)
            glEnd()
            glDisable (GL_SCISSOR_TEST)

        # Save actual console geometry
        self.geometry = (x,y,w,h)

        glColor4f (1, 1, 1, 1)
        glTranslatef (0.0, 0.0, 0.5)
        # Console black border
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE)
        glDisable (GL_TEXTURE_2D)
        glDisable (GL_BLEND)
        glColor4f (0,0,0,0)
        glBegin (GL_QUADS)
        glVertex2f (x, y)
        glVertex2f (x+w, y)
        glVertex2f (x+w, y+h)
        glVertex2f (x, y+h)
        glEnd()        




        glPopAttrib()        
        glMatrixMode (GL_PROJECTION)
        glPopMatrix ()
        glMatrixMode (GL_MODELVIEW)
        glPopMatrix()


    def _framebuffer_on (self, texture):
        """ Activate framebuffer for offscreen rendering """
        
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.framebuffer)
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                                  GL_COLOR_ATTACHMENT0_EXT,
                                  GL_TEXTURE_2D, texture, 0)
        glPushAttrib (GL_ENABLE_BIT | GL_VIEWPORT_BIT)
        glViewport (0, 0, self.width, self.height)
        glMatrixMode (GL_PROJECTION)
        glPushMatrix ()
        glLoadIdentity ()
        glOrtho (0, self.width, 0, self.height, -1, 1)
        glMatrixMode (GL_MODELVIEW)
        glPushMatrix ()
        glLoadIdentity ()
        
        
    def _framebuffer_off (self):
        """ Deactivate framebuffer """

        glMatrixMode (GL_PROJECTION)
        glPopMatrix ()
        glMatrixMode (GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()
        glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0)


    def refresh (self):
        """ Refresh console texture """
        
        # If there is something already bound, just return
#        if glGetInteger (GL_FRAMEBUFFER_BINDING_EXT):
#            return
        if not self.active:
            return

        # If viewport is too small, just return
        h = 5*self.font.size[1] + 2*self.border
        w = 5*self.font.size[0] + 2*self.border   
        if self.width < w or self.height < h:
            return
            
        self._framebuffer_on (self.tex_id)
        r,g,b,a = self.bg_color
        glClearColor (r,g,b,a)
        glClear(GL_COLOR_BUFFER_BIT)

        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL)
        glEnable (GL_TEXTURE_2D)
        glDisable (GL_LIGHTING)
        glEnable (GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        # Actual console surface size
        border = self.border
        h = self.lines*self.font.size[1] + 2*border
        w = self.columns*self.font.size[0] + 2*border       
        x = border + (self.width-w)/2
        y = border + (self.height-h)/2
        glViewport (x, y, w-2*border, h-2*border)
        glMatrixMode (GL_PROJECTION)
        glPushMatrix ()
        glLoadIdentity ()
        glOrtho (0, w-2*border, 0, h-2*border, -1, 1)
        glMatrixMode (GL_MODELVIEW)
        glPushMatrix ()
        glLoadIdentity ()


        # Get displayable lines 
        lines = []
        l = len(self.buffer)
        if l < self.lines:
            for line in self.buffer:
                lines.append (line)
        else:
            a = max(l-self.scroll-self.lines,0)
            b = a + self.lines
            for line in self.buffer[a:b]:
                lines.append (line)
        lines.reverse()
        i = max (self.lines-len(lines),0)

        # Display lines
        r, g, b = self.fg_color
        for line in lines:
            glPushMatrix ()
            glTranslate (0, i*self.font.size[1],0)
            glColor4f (r,g,b,1)
            glListBase (self.font.base)
            self.font.write (line)
            glPopMatrix ()
            i = i + 1

        glMatrixMode (GL_PROJECTION)
        glPopMatrix ()
        glMatrixMode (GL_MODELVIEW)
        glPopMatrix()
        self._framebuffer_off()        

