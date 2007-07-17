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
""" Font class

    Font class allow fast display of text by using truetype font, textures and
    display lists. Note that any truetype font is converted into a fixed width
    font.
"""

import os.path
import gc
import Image, ImageFont
import OpenGL.GL as GL
from glpython.data import datadir

#____________________________________________________________________Font class
class Font:
    """ Font class """
    
    #______________________________________________________________________init
    def __init__ (self, size=12):
        """ Load & initialize font """

        self.allocated = False
        self.size = size
        font_mono   = \
            ImageFont.truetype (os.path.join(datadir(),'sans.ttf'), size)
        font_bold   = \
            ImageFont.truetype (os.path.join(datadir(),'bold.ttf'), size)        
        font_italic = \
            ImageFont.truetype (os.path.join(datadir(),'italic.ttf'), size)       

        
        self.base = GL.glGenLists (3*128)
        self.textures = [None] * 3 * 128
        
        # Compute size in order to have a fixed size font
        max_width, mean_width = 0, 0
        max_height, mean_height = 0, 0
        for i in xrange (32,128):
            glyph = font_mono.getmask (chr (i))
            glyph_width, glyph_height = glyph.size
            max_width = max (max_width, glyph_width)
            mean_width += glyph_width
            max_height = max (max_height, glyph_height)
            mean_height += glyph_height
        mean_width  = int (mean_width/float(128-32))
        mean_height = int (mean_height/float(128-32))

        self.glyph_size = (mean_width, mean_height)
        advance = mean_width
        for i in xrange (128):
            self._make_letters (font_mono, 0, i, advance)
        for i in xrange (128):
            self._make_letters (font_bold, 128, i, advance)
        for i in xrange (128):
            self._make_letters (font_italic, 256, i, advance)

        self.allocated = True
        del font_mono, font_bold, font_italic
        gc.collect()
        
        self.colors = {'black': (0,0,0), 'red':     (1,0,0),
                       'green': (0,1,0), 'yellow':  (1,1,0),
                       'blue':  (0,0,1), 'magenta': (.625,.125,.937),
                       'cyan':  (0,1,1), 'white':   (1,1,1)}

    #____________________________________________________________________render
    def render (self, text, markup = []):
        """ Render text """

        base = 0
        if not text: return
        if markup:
            for m in markup:
                if 'background' in m:
                    color = GL.glGet (GL.GL_CURRENT_COLOR)
                    c = m.split()[0]
                    w = len(text)*self.glyph_size[0]
                    h = self.glyph_size[1]
                    GL.glColor (self.colors[c])
                    GL.glDisable (GL.GL_TEXTURE_2D)
                    GL.glBegin (GL.GL_QUADS)
                    GL.glVertex2f (0, 0)
                    GL.glVertex2f (w, 0)
                    GL.glVertex2f (w, h)
                    GL.glVertex2f (0, h)
                    GL.glEnd()
                    GL.glEnable (GL.GL_TEXTURE_2D)
                    GL.glColor (color)
                elif 'foreground' in m:
                    c = m.split()[0]
                    GL.glColor (self.colors[c])
                if 'bold' in m:
                    base += 128
                elif 'faint' in m:
                    base += 256
        GL.glListBase (self.base+base)
        GL.glCallLists (text)

    #__________________________________________________________________ __del__
    def __del__ (self):
        """ delete """
        
        if self.allocated:
            GL.glDeleteLists (self.base, 3*128)
            for tex_id in self.textures:
                GL.glDeleteTextures (tex_id)
            self.list_base = None
            self.allocated = False
        return

    #____________________________________________________________ _make_letters
    def _make_letters (self, ft, base, ch, advance):
        """  makes one letter (ch) and stores it in a display list """

        def next_p2 (num):
            rval = 1
            while (rval<num):
                rval <<= 1
            return rval
        
        glyph = ft.getmask (chr (ch))
        glyph_width, glyph_height = glyph.size 
        width = next_p2 (glyph_width + 1)
        height = next_p2 (glyph_height + 1)
        expanded_data = ""
        for j in xrange (height):
            for i in xrange (width):
                if i >= glyph_width or j >= glyph_height:
                    value = chr(0)
                    expanded_data += value
                    expanded_data += value
                else:
                    v = glyph.getpixel ((i,j))/256.0
                    vc = chr(int(v*256))
                    #vc = chr (int (256*(v*v)))
                    expanded_data += vc
                    expanded_data += vc

        tex_id = GL.glGenTextures (1)
        GL.glBindTexture (GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameterf (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D (GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                         GL.GL_LUMINANCE_ALPHA, GL.GL_UNSIGNED_BYTE,
                         expanded_data)
        expanded_data = None

        GL.glNewList (self.base + base + ch, GL.GL_COMPILE)
        if ch == ord (" "):
            glyph_advance = glyph_width
            GL.glTranslatef (advance, 0, 0)
            GL.glEndList()
        else:
            GL.glBindTexture (GL.GL_TEXTURE_2D, tex_id)
            GL.glPushMatrix()
            x = float (glyph_width) / float (width)
            y = float (glyph_height) / float (height)            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f (0,0), GL.glVertex2f (0,glyph_height)
            GL.glTexCoord2f (0,y), GL.glVertex2f (0,0)
            GL.glTexCoord2f (x,y), GL.glVertex2f (glyph_width,0)
            GL.glTexCoord2f (x,0), GL.glVertex2f (glyph_width, glyph_height)
            GL.glEnd()
            GL.glPopMatrix ()
            GL.glTranslatef (advance, 0, 0)
            GL.glEndList()
        return glyph.size
