#!/usr/bin/env python

#   Copyright (c) 2007 Nicolas P. Rougier
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   $Id$
#_________________________________________________________________documentation
""" Terminal class

    This class implements rendering functions for the StringTerminal class
    using OpenGL. Input and output buffers are actually rendered into a texture
    using a framebuffer object and is then displayed onto a regular quad using
    ortho mode. The key_press event handler needs to be called each time a key
    is pressed and then the render function has to be called.
"""

import sys, os
import os.path
import Image
import OpenGL.GL as GL
import OpenGL.GL.EXT.framebuffer_object as GL_EXT
#from glpython.data import datadir
from font import Font
from string_terminal import StringTerminal
from shell import Shell
from ishell import Shell as IShell
from io import Infile, Outfile

#________________________________________________________________class Terminal
class Terminal (StringTerminal):
    """ Terminal class """

    #_________________________________________________________________ __init__
    def __init__(self, namespace = {}, shell='ipython',
                 paint_func=None, idle_func=None):
        """ Initialization """
        
        StringTerminal.__init__(self)
        
        self.paint_func = paint_func
        self.idle_func = idle_func
        self.stdout = Outfile (self, sys.stdout.fileno(), self.write_stdout)
        self.stderr = Outfile (self, sys.stderr.fileno(), self.write_stderr)
        self.stdin = Infile (self, sys.stdin.fileno(), self.write_stdout)
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        self.sys_stdin  = sys.stdin
        self.input_mode = False

        if shell == 'python':
            self.shell = Shell (self, namespace)
        else:
            self.shell = IShell (self, namespace)        
                
        self.position = (0,0)
        self.size = (1.0,1.0)
        self.bg_color = (1,1,.85,.75)
        self.fg_color = (0,0,0)
        self.border = 2
        self.default_font_size = 12
        self.visible = True
        self.active = True
        self.depth = 99
        self.focus = False
        self.has_border = True

        # Internal
        self.scroll = 0
        self.font = None
        self.width = 0
        self.height = 0
        self.framebuffer = 0
        self.tex_id = 0
        self.tex_w = 0
        self.tex_h = 0
        self.tex_st = (0,0)
        self.geometry = 0,0,5,5

    #______________________________________________________________write_stdout
    def write_stdout (self, line):
        """ write on stdout in blue """
        self.write ('\033[34m' + line + '\033[00m', self.output_buffer)

    #______________________________________________________________write_stderr
    def write_stderr (self, line):
        """ write on stderr in red """
        self.write ('\033[31m' + line + '\033[00m', self.output_buffer)

    #_________________________________________________________________key_press
    def key_press (self, key):
        """ key press event """
        
        self.scroll = 0
        
        if key == 'control-l':
            self.clear()
            self.dirty = True
            return
        elif key == 'control-v' and os.path.exists('/usr/bin/xsel'):
            try:
                s = os.popen('/usr/bin/xsel').read()
            except:
                return
            if s:
                for c in s:
                    self.key_press (c)

        try:
            result = self.getc (key)
            self.dirty = True
        except KeyboardInterrupt:
            self.read_status = False
            self.shell.cmd = ''
            self.shell.prompt()
            self.dirty = True
            return

        if self.input_mode:
            return
        
        if result:
            self.input_line = self.rl.line
            self.write('\n')
            sys.stdout, sys.stderr = self.stdout, self.stderr
            sys.stdin = self.stdin            
            self.shell.eval (self.input_line)
            sys.stdout, sys.stderr = self.sys_stdout, self.sys_stderr
            sys.stdin = self.sys_stdin
            
    #______________________________________________________________set_position
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
        self.resize()
        self.dirty = True

    #__________________________________________________________________set_size
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
        self.resize()
        self.dirty = True

    #_____________________________________________________________set_char_size
    def set_char_size (self, w, h):
        """ Set character console size
        
            Console character size is specified terms of lines and colums.
        """
        
        if not self.font:
            return
        self.size = (w*self.font.glyph_size[0]+2*self.border, 
                     h*self.font.glyph_size[1]+2*self.border)
        self.resize()
        self.dirty = True

    #______________________________________________________________set_bg_color
    def set_bg_color (self, color):
        """ Set console background color
        
            Console color must be specified as a (red,green,blue,alpha) tuple
            of float between 0 and 1. An alpha value of 0 means a fully
            transparent background while a value of 1 means a fully opaque
            background.
        """
        
        self.bg_color = color
        self.dirty = True

    #______________________________________________________________set_fg_color
    def set_fg_color (self, color):
        """ Set console foreground (text) color
        
        Console color must be specified as a (red,green,blue) tuple of
        float between 0 and 1.
        """
        
        self.fg_color = color
        self.dirty = True

    #_____________________________________________________________set_font_size
    def set_font_size (self, size):
        """ Set console font size
        
            Font size must be a non null integer
        """
        
        if size == 0:
            size = self.default_font_size
        self.font = Font (size)
        self.reshape (force=True)

    #_______________________________________________________________scroll_down
    def scroll_down (self, amount=5):
        """ Scroll console down """
        
        self.scroll -= amount
        self.dirty = True

    #_________________________________________________________________scroll_up
    def scroll_up (self, amount=5):
        """ Scroll console up """
        
        self.scroll +=  amount
        self.dirty = True

    #______________________________________________________________________init
    def init (self):
        """ Initialize console
        
            Initialize function must be called as soon as an OpenGL context is
            available.
        """
        
        self.font = Font (self.default_font_size)
        self.set_fg_color (self.fg_color)
        self.dirty = True
        
    #_______________________________________________________________check_focus
    def check_focus (self, x, y):
        """ Check if console has focus """
        
        xx,yy,w,h = self.geometry
        if x > xx and x < (xx+w) and y > yy and y < (yy+h):
            self.focus = True
        else:
            self.focus = False

    #___________________________________________________________________reshape
    def resize (self, w=None, h=None, force=False):
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
      	    viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
            w = viewport[2] - viewport[0]
            h = viewport[3] - viewport[1]

        max_w = max_h = GL.glGetIntegerv (GL.GL_MAX_TEXTURE_SIZE)
        
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

        # Check if we're below card specs
        if width > max_w:
            width = max_w
        if height > max_h:
            height = max_h

        # Now we adjust size according to font glyph size
        width  = max(width, 2*self.border+self.font.glyph_size[0])
        height = max(height,2*self.border+self.font.glyph_size[1])

        # Computes console position
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
        self.x, self.y = x,y
        
        # No change needed
        if self.width == width and self.height == height and not force:
            return
        
        # Update actual size
        self.width = width
        self.height = height

        # Compute console size in characters
        self.lines   = (self.height-2*self.border)/self.font.glyph_size[1]
        self.columns = (self.width-2*self.border)/self.font.glyph_size[0]

        # Refresh output line
        if self.read_status:
            self.input_buffer = []
            self.write (self.prompt+self.rl.line, self.input_buffer)
            c = self.rl.cursor + self.prompt_len    
            self.cursor = (c%self.columns, c/self.columns)

        # If framebuffer is already bound, we keep the old texture
        # if GL.glGetIntegerv (GL.GL_FRAMEBUFFER_BINDING_EXT):
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
            GL.glDeleteTextures (self.tex_id)
            GL_EXT.glDeleteFramebuffersEXT (1, [self.framebuffer])
        image = Image.new ("RGB", (self.tex_w, self.tex_h), (0, 0, 0))
        bits = image.tostring("raw", "RGBX", 0, -1)

        self.framebuffer = int(GL_EXT.glGenFramebuffersEXT (1)) #[0]
        self.tex_id = GL.glGenTextures (1)
        GL.glBindTexture (GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexParameteri (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D (GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                         self.tex_w, self.tex_h, 0,
                         GL.GL_RGB, GL.GL_UNSIGNED_BYTE, bits)
        image = None
        bits = None
        self.dirty = True
        self.refresh()

    #____________________________________________________________________render
    def render (self):
        """ Render console
        
            Rendering is straightforward since what need to be displayed is
            contained within a texture.
        """

        mode = GL.glGetIntegerv (GL.GL_RENDER_MODE)
        if (mode == GL.GL_SELECT):
            return

        if self.dirty:
            self.refresh()
            self.dirty = False

      	viewport = GL.glGetIntegerv (GL.GL_VIEWPORT)
        w = viewport[2] - viewport[0]
        h = viewport[3] - viewport[1]
        
        # "2D" projection mode
        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glPushMatrix ()
        GL.glLoadIdentity ()
        GL.glOrtho (0, w, 0, h, -100, 100)
        GL.glMatrixMode (GL.GL_MODELVIEW)
        GL.glPushMatrix ()
        GL.glLoadIdentity ()
        
        GL.glPushAttrib (GL.GL_ENABLE_BIT | GL.GL_VIEWPORT_BIT)
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable (GL.GL_TEXTURE_RECTANGLE_ARB)        
        GL.glEnable (GL.GL_BLEND)
        GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
        
        # Just for fun
        # GL.glRotate (2.5,0,0,1)

        # Computes console position
        x, y = self.x, self.y
        w,h = self.width, self.height
        s,t = self.tex_st

        # Console
        GL.glEnable (GL.GL_TEXTURE_2D)
        GL.glBindTexture (GL.GL_TEXTURE_2D, self.tex_id)
        GL.glColor ( (1,1,1,1) )
        GL.glTranslatef (0.0, 0.0, self.depth)
        GL.glBegin (GL.GL_QUADS)
        GL.glTexCoord2f(0, 0), GL.glVertex2f (x, y)
        GL.glTexCoord2f(s, 0), GL.glVertex2f (x+w, y)
        GL.glTexCoord2f(s, t), GL.glVertex2f (x+w, y+h)
        GL.glTexCoord2f(0, t), GL.glVertex2f (x, y+h)
        GL.glEnd()

        # Save actual console geometry
        self.geometry = (x,y,w,h)
        GL.glTranslatef (0.0, 0.0, 0.5)

        # Console black border
        if self.has_border:
            GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glDisable (GL.GL_TEXTURE_2D)
            GL.glDisable (GL.GL_BLEND)
            GL.glColor4f (0,0,0,1)
            GL.glBegin (GL.GL_QUADS)
            GL.glVertex2f (x, y)
            GL.glVertex2f (x+w-1, y)
            GL.glVertex2f (x+w-1, y+h-1)
            GL.glVertex2f (x, y+h-1)
            GL.glEnd()        

        GL.glPopAttrib()    
        GL.glDisable (GL.GL_TEXTURE_2D)    
        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glPopMatrix ()
        GL.glMatrixMode (GL.GL_MODELVIEW)
        GL.glPopMatrix()

    #___________________________________________________________________refresh
    def refresh (self):
        """ Refresh console texture """
        
        if not self.active:
            return

        GL_EXT.glBindFramebufferEXT (GL_EXT.GL_FRAMEBUFFER_EXT,
                                     self.framebuffer)
        GL_EXT.glFramebufferTexture2DEXT (GL_EXT.GL_FRAMEBUFFER_EXT,
                                          GL.GL_COLOR_ATTACHMENT0_EXT,
                                          GL.GL_TEXTURE_2D, self.tex_id, 0)
        GL.glPushAttrib (GL.GL_ENABLE_BIT | GL.GL_VIEWPORT_BIT)
        GL.glViewport (0, 0, self.width, self.height)

        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glPushMatrix ()
        GL.glLoadIdentity ()
        GL.glOrtho (0, self.width, 0, self.height, -1, 1)
        GL.glMatrixMode (GL.GL_MODELVIEW)
        GL.glPushMatrix ()
        GL.glLoadIdentity ()

        r,g,b,a = self.bg_color
        GL.glClearColor (r,g,b,a)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glPolygonMode (GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable (GL.GL_TEXTURE_RECTANGLE_ARB)
        GL.glEnable (GL.GL_TEXTURE_2D)
        GL.glDisable (GL.GL_LIGHTING)
        GL.glEnable (GL.GL_BLEND)
        GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glTexEnvf (GL.GL_TEXTURE_ENV,
                      GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)

        GL.glEnable (GL.GL_SCISSOR_TEST)
        GL.glScissor (self.border, self.border,
                      self.width-2*self.border,self.height-2*self.border)

        # Actual rendering of lines
        if self.scroll < 0:
            self.scroll = 0
        elif ((len (self.output_buffer)-self.lines+1) > 0 and 
              self.scroll > (len (self.output_buffer)-self.lines+1)):
            self.scroll = len (self.output_buffer)-self.lines+1

        if self.output_buffer and self.output_buffer[-1] == [[[], '']]:
            b = self.output_buffer[:-1] + self.input_buffer
        else:
            b = self.output_buffer + self.input_buffer

        if len(b) <= self.lines:
            self.scroll = 0
            
        start = max (-self.lines-1-self.scroll, -len(b))
        end = -self.scroll
        if end >= -1:
            end = None


        # If we have less than self.lines to display, we use upper left corner
        # as the origin, while if we have self.lines to display, we use bottom
        # left corner
        if len(b[start:end]) >= self.lines:
            y = self.lines
            dy = 0
        else:
            y = self.lines-1
            dy = (self.height-2*self.border) % self.font.glyph_size[1]
            start -= 1

        for segments in b[start:end]:
            GL.glPushMatrix ()
            GL.glTranslate (self.border,
                            self.border+dy+y*self.font.glyph_size[1],
                            0)
            for markup,line in segments:
                GL.glColor (self.fg_color)
                self.font.render (line, markup)            
            GL.glPopMatrix ()
            y -= 1
            
        # Rendering of cursor
        GL.glColor (self.fg_color)
        x = self.cursor[0]
        y = self.lines + start + len(self.input_buffer) - self.cursor[1]
        x = x*self.font.glyph_size[0]+2
        y = y*self.font.glyph_size[1] + dy
        GL.glDisable (GL.GL_TEXTURE_2D)
        GL.glPushMatrix ()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex2f (x,y)
        GL.glVertex2f (x,y+self.font.glyph_size[1])
        GL.glEnd()
        GL.glPopMatrix()

        GL.glDisable (GL.GL_SCISSOR_TEST)
        GL.glMatrixMode (GL.GL_PROJECTION)
        GL.glPopMatrix ()
        GL.glMatrixMode (GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glPopAttrib()
        GL_EXT.glBindFramebufferEXT (GL_EXT.GL_FRAMEBUFFER_EXT, 0)

