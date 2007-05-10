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

""" Text buffer

This defines a text buffer where only the last line can be edited. All
positions are given as integers (positive or negative) and relates to the
current edited line only. It is also possible to use the prompt method for
displaying a prompt that won't be considered to be part of the edited line.
Color and font markers are handled and made invisible for the user. 

"""

from font import *
from font import _FontMarker, _ColorMarker


class TextBuffer:
    """ Text buffer class """
    
    def __init__ (self):
        """ Initialize the buffer """

        self.buffer = ['',]
        self.cursor = -1
        self.protect = 0
        self.dirty = True
        self.scroll = 0
        self.buffer[-1] += Cursor

    def insert (self, text, position=-1, color=None, font=None):
        """ Insert text at position """

        text = colorify (text, color)
        text = fontify (text, font)

        y = len(self.buffer)-1
        self.buffer[y] = self.buffer[y].replace (Cursor,'')

        x = self.index (position)
        last_color = None
        last_font  = None
        text = self.buffer[y][:x] + text + self.buffer[y][x:]
        self.buffer[y] = ''

        for c in text:
            if is_color(c):
                if c == DefaultColor: last_color = None
                else:                 last_color = c
            elif is_font(c):
                if c == DefaultFont:  last_font = None
                else:                 last_font = c
            self.buffer[y] += c
            if c == '\n':
                l = ""
                if last_color:
                    l += last_color
                if last_font:
                    l += last_font + _FontMarker
                self.buffer.insert(y+1, l)
                self.protect = 0
                y += 1
        self.dirty = True
        self.scroll = 0
        i = self.index(self.cursor)
        self.buffer[-1] = self.buffer[-1][:i] +Cursor+ self.buffer[-1][i:]


    def write (self, text, color=None, font=None):
        """ Write text at current cursor position """
        
        self.insert (text, self.cursor, color, font)
        self.scroll = 0


    def replace (self, text,  color=None, font=None):
        """ Replace current edited line with text """
        
        self.buffer[-1] = self.buffer[-1].replace (Cursor,'')
        self.buffer[-1] = self.buffer[-1][0:self.protect]
        self.cursor = -1
        self.insert (text, -1)
        self.dirty = True
        self.scroll = 0


    def clear (self):
        """ Clear buffer """
        
        self.buffer = self.buffer[-1:]
        self.dirty = True
        self.scroll = 0


    def delete (self, start, end):
        """ Delete buffer from start to end """

        if len(self.buffer) == 0:
            return

        self.buffer[-1] = self.buffer[-1].replace (Cursor,'')
        s = self.index(start)
        e = self.index(end)
        if s > e:
            text = self.buffer[-1][:e] + self.buffer[-1][s:]
        else:
            text = self.buffer[-1][:s] + self.buffer[-1][e:]
        self.buffer[-1] = text
        self.dirty = True
        self.scroll = 0
        i = self.index(self.cursor)
        self.buffer[-1] = self.buffer[-1][:i] +Cursor+ self.buffer[-1][i:]


    def move (self, iterations):
        """ Move cursor relative to current position """

        if iterations < 0:
            d = -1
        elif iterations > 0:
            d = +1
        else:
            return

        i = self.index (self.cursor)
        if (d>0 and self.cursor == -1) or (d<0 and i == self.protect):
            return
        for i in range(abs(iterations)):
            self.cursor += d
            i = self.index (self.cursor)
            if (d>0 and self.cursor == -1) or (d<0 and i == self.protect):
                break
        self.dirty = True
        self.scroll = 0
        self.buffer[-1] = self.buffer[-1].replace (Cursor,'')
        i = self.index(self.cursor)
        self.buffer[-1] = self.buffer[-1][:i] +Cursor+ self.buffer[-1][i:]


    def move_start (self):
        """ Move cursor to start of line """
        
        if self.buffer[-1]:
            self.move (-len(self.buffer[-1]))
        self.dirty = True
        self.scroll = 0


    def move_end (self):
        """ Move cursor to end of line """
        
        self.cursor = -1
        self.dirty = True
        self.scroll = 0
        self.buffer[-1] = self.buffer[-1].replace (Cursor,'')
        self.buffer[-1] += Cursor


    def current_line (self, index=-1):
        """ Return the current edited line """

        line = ''
        for c in self.buffer[-1][self.protect:]:
            if c >= ' ' and c < chr(128):
                line += c
        return line


    def index (self, position=None):
        """ Return the positive index corresponding to a position """

        if position == None:
            position = -1
        line = self.buffer[-1]

        if position >= 0:
            x = self.protect
            for x in xrange(len(line)):
                if position == 0:
                    return max(x,self.protect)
                elif line[x] >= ' ':
                    position -= 1
            return len(line)
        else:
            x = 0
            for x in xrange(len(line)):
                if position == -1:
                    return max(len(line)-x, self.protect)
                elif line[len(line)-1-x] >= ' ':
                    position += 1
            return self.protect

