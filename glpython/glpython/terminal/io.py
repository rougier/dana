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
""" Fake I/O """

class Outfile:
    """
    A fake output file object.  It sends output to the console and if
    asked for a file number, returns one set on instance creation.
    """
    
    def __init__(self, console, fn, write):
        self.fn = fn
        self.console = console
        self.write = write
    def close(self): pass
    flush = close
    def fileno(self):    return self.fn
    def isatty(self):    return False
    def read(self, a):   return ''
    def readline(self):  return ''
    def readlines(self): return []
    def write(self, s):
        self.write (s, delayed=True)
    def writelines(self, l):
        for s in l:
            self.write (s, delayed=True)
    def seek(self, a):   raise IOError, (29, 'Illegal seek')
    def tell(self):      raise IOError, (29, 'Illegal seek')
    truncate = tell


class Infile:
    """
    A fake input file object.  It receives input from the console and if
    asked for a file number, returns one set on instance creation.
    """
    
    def __init__(self, console, fn):
        self.fn = fn
        self.console = console
    def close(self): pass
    flush = close
    def fileno(self):    return self.fn
    def isatty(self):    return False
    def read(self, a):   return self.readline()
    def readline(self):
        self.console.input_mode = True
        while self.console.input_mode:
            self.console.idle()
            self.console.paint()
        if not self.console.interrupt:
            s = self.console.input
        else:
            s = None
            self.console.interrupt = False
            raise KeyboardInterrupt
        self.console.input = ''
        return s
    def readlines(self): return []
    def write(self, s):  return None
    def writelines(self, l): return None
    def seek(self, a):   raise IOError, (29, 'Illegal seek')
    def tell(self):      raise IOError, (29, 'Illegal seek')
    truncate = tell
