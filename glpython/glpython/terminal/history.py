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

""" Basic history

"""

import os, atexit

class History:
    """ Basic command history class """
    
    def __init__ (self, filename=None):
        """ Initializes history """
        if not filename:
            self.filename = os.path.expanduser("~/.pyhistory")
        else:
            self.filename = filename
        self.history = ['']
        self.open()
        self.position = len (self.history)-1
        atexit.register(self.save)

    def prev (self, current):
        """ Get previous command in history """
        
        if self.position > 0:
            l = current
            if len(l) > 0 and l[0] == '\n': l = l[1:]
            if len(l) > 0 and l[-1] == '\n': l = l[:-1]
            if self.position > 0:
                if self.position == (len(self.history)-1):
                    self.history[len(self.history)-1] = l
                self.position = self.position - 1
                return self.history[self.position]
        return current


    def next (self, current):
        """ Get next command in history """
        
        if self.position < len(self.history) - 1:
            self.position = self.position + 1
            return self.history[self.position]
        return current


    def append (self, line):
        """ Append a new command to history """
        
        if not len(line):
            return
        if line in self.history:
            index = self.history.index(line)
            del self.history[index]
        self.history[-1] = line
        self.position = len(self.history)
        self.history.append('')

    def open (self):
        """ Open history file """
        
        try:
            file = open (self.filename)
            self.history = []
            for l in file:
                self.history.append(l[:-1])
            self.history.append('')
            self.position = len(self.history)-1
            file.close()
        except:
            pass

    def save (self):
        """ Save history to a file """
        
        file = open (self.filename, 'w')
        for l in self.history[-256:]:
            if len(l) > 0:
                file.write(l+'\n')
        file.close()


    def __repr__(self):
        """ History representation """
        
        return self.history.__repr__()
