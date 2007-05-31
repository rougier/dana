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

""" Infile/Outfile class 

"""

class Outfile:
    """ Outfile class

        A fake output file object.  It sends output to the terminal and if
        asked for a file number, returns one set on instance creation.
    """
    
    def __init__(self, terminal, fn, write_func):
        self.fn = fn
        self.terminal = terminal
        self.write_func = write_func
    def close(self): pass
    flush = close
    def fileno(self):    return self.fn
    def isatty(self):    return False
    def read(self, a):   return ''
    def readline(self):  return ''
    def readlines(self): return []
    def write(self, s):
        self.write_func (s)
    def writelines(self, l):
        for s in l:
            self.write_func (s)
    def seek(self, a):   raise IOError, (29, 'Illegal seek')
    def tell(self):      raise IOError, (29, 'Illegal seek')
    truncate = tell


class Infile:
    """ Infile class

        A fake input file object.  It receives input from the terminal and if
        asked for a file number, returns one set on instance creation.
    """
    
    def __init__(self, terminal, fn, write_func):
        self.fn = fn
        self.terminal = terminal
        self.write_func = write_func
    def close(self): pass
    flush = close
    def fileno(self):    return self.fn
    def isatty(self):    return False
    def read(self, a):   return self.readline()
    def readline(self):
        self.terminal.read ('')
        self.terminal.dirty = True
        self.terminal.input_mode = True
        while self.terminal.read_status:
            self.terminal.idle_func()
            self.terminal.paint_func()
        self.write_func ('\n')
        self.terminal.dirty = True
        self.terminal.input_mode = False
        return self.terminal.rl.line
    def readlines(self): return []
    def write(self, s):  return None
    def writelines(self, l): return None
    def seek(self, a):   raise IOError, (29, 'Illegal seek')
    def tell(self):      raise IOError, (29, 'Illegal seek')
    truncate = tell
