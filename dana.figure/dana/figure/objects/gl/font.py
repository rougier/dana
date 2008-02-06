#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# DANA 
# Copyright (C) 2006-2007  Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
#------------------------------------------------------------------------------
""" Font class

    Font class allow fast display of text by using truetype font, textures and
    display lists. Note that any truetype font is converted into a fixed width
    font.
"""

import os.path
import gc
import Image, ImageFont
import OpenGL.GL as GL
import OpenGL.GLU as GLU


class Font:
    """ Font class """
    
    def __init__ (self, filename):
        """ Load & initialize font """

        self.allocated = False
        font = ImageFont.truetype (filename, 50)
        self.base = GL.glGenLists (128)
        self.textures = [None] * 128
        self.char_size = [0,0] * 128
        for i in xrange (128):
            self.char_size[i] = self.make_letter (font, i)
        self.allocated = True
        del font
        gc.collect()

    def render (self, text):
        """ Render text """

        if not text: return
        GL.glListBase (self.base)
        GL.glCallLists (text)

    def extents (self, text):
        """ Compute text extents """

        w,h = 0,0
        for c in text:
            s = self.char_size[ord(c)]
            if s[1] > h:
                h = s[1]
            w += s[0]
        return w,h

    def __del__ (self):
        """ delete """
        
        if self.allocated:
            GL.glDeleteLists (self.base, 128)
            for tex_id in self.textures:
                GL.glDeleteTextures (tex_id)
            self.list_base = None
            self.allocated = False
        return

    def make_letter (self, ft, ch):
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
        GL.glTexImage2D (GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                         GL.GL_LUMINANCE_ALPHA, GL.GL_UNSIGNED_BYTE,
                         expanded_data)
        GL.glTexParameterf (GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                            GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameterf (GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP)
        GL.glTexParameterf (GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP)
        GLU.gluBuild2DMipmaps (GL.GL_TEXTURE_2D, GL.GL_RGBA, width, height,
                               GL.GL_LUMINANCE_ALPHA, GL.GL_UNSIGNED_BYTE,
                               expanded_data)
        expanded_data = None
        GL.glNewList (self.base + ch, GL.GL_COMPILE)
        if ch == ord (" "):
            GL.glTranslatef (glyph_width, 0, 0)
            GL.glEndList()
        else:
            GL.glBindTexture (GL.GL_TEXTURE_2D, tex_id)
            GL.glPushMatrix()
            x = float (glyph_width) / float (width)
            y = float (glyph_height) / float (height) 
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f (0,0), GL.glVertex2f (0,glyph_height)
            GL.glTexCoord2f (0,y), GL.glVertex2f (0,0)
            GL.glTexCoord2f (x,y), GL.glVertex2f (glyph_width,0)
            GL.glTexCoord2f (x,0), GL.glVertex2f (glyph_width, glyph_height)
            GL.glEnd()
            GL.glPopMatrix ()
            GL.glTranslatef (glyph_width, 0, 0)
            GL.glEndList()
        return glyph.size
