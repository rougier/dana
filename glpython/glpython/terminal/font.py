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

""" OpenGL font utilities

The Font class allows to display truetype text onto an OpenGL scene.

"""

import os.path
import Image, ImageFont
import OpenGL.GL as GL
from glpython.data import datadir


# Font color
_ColorMarker = chr(10)
_ColorStart  = chr(128)
Black        = chr(ord(_ColorStart)+1)
Red          = chr(ord(_ColorStart)+2)
Green        = chr(ord(_ColorStart)+3)
Brown        = chr(ord(_ColorStart)+4)
Blue         = chr(ord(_ColorStart)+5)
Purple       = chr(ord(_ColorStart)+6)
Cyan         = chr(ord(_ColorStart)+7)
LightGray    = chr(ord(_ColorStart)+8)
DarkGray     = chr(ord(_ColorStart)+9)
DarkRed      = chr(ord(_ColorStart)+10)
SeaGreen     = chr(ord(_ColorStart)+11)
Yellow       = chr(ord(_ColorStart)+12)
LightBlue    = chr(ord(_ColorStart)+13)
MediumPurple = chr(ord(_ColorStart)+14)
LightCyan    = chr(ord(_ColorStart)+15)
White        = chr(ord(_ColorStart)+16)
DefaultColor = Black

# Font style
_FontMarker = chr(19)
_FontStart  = chr(150)
Regular     = chr(ord(_FontStart)+1)
Bold        = chr(ord(_FontStart)+2)
Italic      = chr(ord(_FontStart)+3)
DefaultFont = Regular


# Cursor marker
Cursor = chr(255)


def is_color (marker):
    """ Test whether a marker is a color marker """
    return marker in [Black, Red, Green, Brown, Blue, Purple, Cyan,
                      LightGray, DarkGray, DarkRed, SeaGreen, Yellow,
                      LightBlue, MediumPurple, LightCyan, White]

def is_font (marker):
    """ Test whether a marker is a font marker """
    return marker in [Regular, Bold, Italic]



def colorify (text, color):
    """ Colorify text with the specified color """
    
    if text=='' or not is_color(color):
        return text
    if text[-1] == '\n':
        return (color + text[:-1].replace ('\n', '\n' + color) + Black + '\n')
    else:
        return (color + text.replace ('\n', '\n' + color) + Black)
    return text


def fontify (text, font):
    """ Fontify text with the specified font """
    
    if not text or not is_font (font):
        return text

    font += _FontMarker
    regular = Regular + _FontMarker  
    if text[-1] == '\n':
        return (font + text[:-1].replace ('\n', '\n'+ font) + regular +'\n')
    else:
        return (font + text.replace ('\n', '\n' + font) + regular)
    return text


class Font:
    """ Font class

    The font class uses the ImageFont library to produce several OpenGL
    textures (one per letter) that are then saved in display lists. The trick
    is to use also "special" display lists that either change the color of
    following letters or change the font by modifying the base of display
    lists. Printing some text onto screen is then straightforward.

    """
    
    def __init__ (self, size=12,
                  filename=None, bold_filename=None, italic_filename=None):
        """ Load & initialize fonts """

        self.m_allocated = False
        self.size = size
        
        default = os.path.join(datadir(),'sans.ttf')
        default_bold = os.path.join(datadir(),'bold.ttf')
        default_italic = os.path.join(datadir(),'italic.ttf')
        normal_ok = False

        normal = None
        bold = None
        italic = None
        if filename:
            try:
                normal = ImageFont.truetype(filename, size)
                bold = normal
                italic = normal
            except:
                normal = ImageFont.truetype (default, size)
                bold = ImageFont.truetype (default_bold, size)
                italic = ImageFont.truetype (default_italic, size)
        else:
            normal = ImageFont.truetype (default, size)
            bold = ImageFont.truetype (default_bold, size)
            italic = ImageFont.truetype (default_italic, size)

        if bold_filename:
            try:
                bold = ImageFont.truetype(bold_filename, size)
            except:
                pass
        if italic:
            try:
                italic = ImageFont.truetype(regular, size)
            except:
                pass
        
        self.base = GL.glGenLists (256*3)
        self.textures = [None] * 256 * 3
        self.regular_w = [0] * 256
        self.bold_w = [0] * 256
        self.italic_w = [0] * 256

        for i in xrange (128):
            size = self._make_letters (normal, i, self.base+0*256,
                                        self.textures, self.regular_w)
            size = self._make_letters (bold, i, self.base+1*256,
                                        self.textures, self.bold_w)
            size = self._make_letters (italic, i, self.base+2*256,
                                        self.textures, self.italic_w)
            if i== 32:
                self.size = size

        for i in xrange(32):
            self._make_void (i)
        for i in xrange(128):
            self._make_void (128+i)


        self._make_color_marker (ord(Black),        (0,0,0))
        self._make_color_marker (ord(Red),          (1,0,0))
        self._make_color_marker (ord(Green),        (0,1,0))
        self._make_color_marker (ord(Brown),        (.645,.164,.164))
        self._make_color_marker (ord(Blue),         (0,0,1))
        self._make_color_marker (ord(Purple),       (.625,.125,.937))
        self._make_color_marker (ord(Cyan),         (0,1,1))
        self._make_color_marker (ord(LightGray),    (.825,.825,.825))
        self._make_color_marker (ord(DarkGray),     (.66,.66,.66))
        self._make_color_marker (ord(DarkRed),      (.543,0,0))
        self._make_color_marker (ord(SeaGreen),     (.18,.543,.34))
        self._make_color_marker (ord(Yellow),       (1,1,0))
        self._make_color_marker (ord(LightBlue),    (.676,.844,.9))
        self._make_color_marker (ord(MediumPurple), (.574,.437,.855))
        self._make_color_marker (ord(LightCyan),    (.875,1,1))
        self._make_color_marker (ord(White),        (1,1,1))

        
        self._make_font_marker (ord(Regular), 0*256)
        self._make_font_marker (ord(Bold), 1*256)
        self._make_font_marker(ord(Italic), 2*256)
        self._make_cursor (ord(Cursor)+0*256)
        self._make_cursor (ord(Cursor)+1*256)
        self._make_cursor (ord(Cursor)+2*256)

        self.m_allocated = True
        del normal, bold, italic
        return


    def write (self, string):
        """  Display text """
        
        if not string:
            return
        strings = string.split(_FontMarker)
        for s in strings:
            if s:
                GL.glCallLists (s)
        return


    def __del__ (self):
        """  """
        
        if self.m_allocated:
            GL.glDeleteLists (self.base, 3*256)
            for tex_id in self.textures:
                GL.glDeleteTextures (tex_id)
            self.list_base = None
            self.m_allocated = False
        return


    def _make_cursor (self, list_id):
        """ Build display list for cursor """
        
        GL.glNewList (self.base + list_id, GL.GL_COMPILE)
        GL.glPushAttrib (GL.GL_ENABLE_BIT)
        GL.glDisable (GL.GL_TEXTURE_2D)
        GL.glDisable (GL.GL_BLEND)
        GL.glBegin (GL.GL_LINES)
        GL.glVertex2f (0, self.size[1]*.2)
        GL.glVertex2f (0, self.size[1]*1.1)
        GL.glEnd ()
        GL.glPopAttrib ()
        GL.glEndList ()
        

    def _make_color_marker (self, list_id, color):
        """ Build color display list for all 3 styles """
        
        GL.glNewList (self.base + list_id + 0*256, GL.GL_COMPILE)
        GL.glColor (color[0], color[1], color[2], 1)
        GL.glEndList()
        GL.glNewList (self.base + list_id + 1*256, GL.GL_COMPILE)
        GL.glColor (color[0], color[1], color[2], 1)
        GL.glEndList()
        GL.glNewList (self.base + list_id + 2*256, GL.GL_COMPILE)
        GL.glColor (color[0], color[1], color[2], 1)
        GL.glEndList()
        return


    def _make_font_marker (self, list_id, base):
        """ Build font display list for all 3 styles """

        GL.glNewList (self.base + list_id + 0*256, GL.GL_COMPILE)
        GL.glListBase (self.base + base)
        GL.glEndList()
        GL.glNewList (self.base + list_id + 1*256, GL.GL_COMPILE)
        GL.glListBase (self.base + base)
        GL.glEndList()
        GL.glNewList (self.base + list_id + 2*256, GL.GL_COMPILE)
        GL.glListBase (self.base + base)
        GL.glEndList()
        return


    def _make_void (self, list_id):
        """ Build void display list for all 3 styles """

        GL.glNewList (self.base + list_id + 0*256, GL.GL_COMPILE)
        GL.glEndList()
        GL.glNewList (self.base + list_id + 1*256, GL.GL_COMPILE)
        GL.glEndList()
        GL.glNewList (self.base + list_id + 2*256, GL.GL_COMPILE)
        GL.glEndList()
        return


    def _make_letters (self, ft, ch, list_base, tex_base_list, font_w):
        """  """

        def next_p2 (num):
            rval = 1
            while (rval<num):
                rval <<= 1
            return rval
        
        glyph = ft.getmask (chr (ch))
        glyph_width, glyph_height = glyph.size 

        if ch < 32:
            font_w[ch] = 0
        else:
            font_w[ch] = glyph_width

        width = next_p2 (glyph_width + 1)
        height = next_p2 (glyph_height + 1)
        expanded_data = ""
        for j in xrange (height):
            for i in xrange (width):
                if i >= glyph_width or j >= glyph_height:
                    value = chr (0)
                    expanded_data += value
                    expanded_data += value
                else:
                    value = chr(glyph.getpixel ((i,j)))
                    expanded_data += value
                    expanded_data += value

        tex_id = GL.glGenTextures (1)
        tex_base_list [ch] = tex_id
        GL.glBindTexture (GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameterf (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf (GL.GL_TEXTURE_2D,
                            GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D (GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                         GL.GL_LUMINANCE_ALPHA, GL.GL_UNSIGNED_BYTE,
                         expanded_data)
        expanded_data = None

        GL.glNewList (list_base + ch, GL.GL_COMPILE)
        if ch == ord (" "):
            glyph_advance = glyph_width
            GL.glTranslatef (glyph_advance, 0, 0)
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
