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
""" Readline emulation using strings

    The Readline class implements a GNU readline emulation that is completely
    disconnected from any terminal. User has to manually add character using
    the getc function until the line is done edited (when getc returns true)
    and is responsible for the actual display of the current line. A completer
    function may be provided that will be called upon completion.
"""

import os, atexit


#__________________________________________________________class StringReadline
class StringReadline:
    """ StringReadline class """

    #_________________________________________________________________ __init__
    def __init__ (self, completer=None):
        """ Initialization """
        
        self.char = ''
        self.line = ''
        self.copy = ''
        self.kill_buffer = ''
        self.cursor = 0
        self.maxsize = 0
        self.line_ready = False
        self.completer = completer
        self.history = []
        self.history_pos = 0
        self.bindings = {'control-a': self.move_start,
                         chr(1):      self.move_start,
                         'control-b': self.move_char_backward,
                         'control-c': self.keyboard_interrupt,
                         chr(2):      self.move_char_backward,
                         'left':      self.move_char_backward,
                         'control-d': self.delete_char_forward,
                         chr(4):      self.delete_char_forward,
                         'del':       self.delete_char_forward,
                         'control-e': self.move_end,
                         chr(5):      self.move_end,
                         'control-f': self.move_char_forward,
                         chr(6):      self.move_char_forward,
                         'right':     self.move_char_forward,
                         'up':        self.history_up,
                         'down':      self.history_down,
                         'control-k': self.kill_forward,
                         chr(11):     self.kill_forward,
                         'control-y': self.yank,
                         chr(25):     self.yank,
                         'backspace': self.delete_char_backward,
                         'delete':    self.delete_char_forward,
                         'up':        self.history_up,
                         'down':      self.history_down,
                         'tab':       self.complete,
                         '\t':        self.complete,
                         'enter':     self.end_of_line,
                         'return':    self.end_of_line,
                         '\n':        self.end_of_line
                        }
        self.printable = '''0123456789''' \
                         '''abcdefghijklmnopqrstuvwxyz''' \
                         '''ABCDEFGHIJKLMNOPQRSTUVWXYZ''' \
                         '''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ''' \
                         '''\t'''
        self.filename = os.path.expanduser("~/.pyhistory")
        self.read_history_file ()
        atexit.register(self.write_history_file)

    #_____________________________________________________________________start
    def start (self, maxsize=0):
        """ start editing a new line """

        self.cursor = 0
        self.line = ''
        self.maxsize = maxsize
        self.line_ready = False
        if self.history:
            self.history_pos = len(self.history)

    #____________________________________________________________________cancel
    def cancel (self):
        """ Cancel editing """

        self.start()

    #______________________________________________________________________getc
    def getc (self, c):
        """ process char and return True if line is ready """

        self.char = c
        if self.line_ready:
            return self.line_ready
        
        if c in self.bindings.keys():
            if self.bindings[c]():
                return self.line_ready
            
        if c in self.printable:
            if not self.maxsize or len(self.line) < self.maxsize:
                cursor = self.cursor
                self.line = self.line[:cursor] + c + self.line[cursor:]
                self.cursor += 1
        return self.line_ready

    #___________________________________________________________________replace
    def replace (self, line):
        """ replace current line """

        self.line = line
        if self.maxsize and len(self.line) > self.maxsize:
            self.line = self.line[:self.maxsize]
        self.cursor = len(self.line)

    #________________________________________________________keyboard_interrupt
    def keyboard_interrupt (self):
        """ keyboard interrupt """

        self.line_ready = False
        raise KeyboardInterrupt

    #_______________________________________________________________end_of_line
    def end_of_line (self):
        """ line is ready to be processed """

        self.line_ready = True
        if (not self.history or 
                (len(self.line) > 0 and self.line != self.history[-1])):
            self.history.append (self.line)
            self.history_pos = len(self.history)
        return True

    #__________________________________________________________________complete
    def complete (self):
        """ completion """

        if self.completer:
            self.completer (self, self.line)
            return True
        else:
            return False

    #________________________________________________________________move_start
    def move_start (self):
        """ move to the start of the line """

        self.cursor = 0
        return True

    #__________________________________________________________________move_end
    def move_end (self):
        """ move to the end of the line """

        self.cursor = len(self.line)
        return True

    #________________________________________________________move_char_backward
    def move_char_backward (self):
        """ move cursor back one character """

        self.cursor = max (0, self.cursor-1)
        return True

    #_________________________________________________________move_char_forward
    def move_char_forward (self):
        """ move cursor forward one character """

        self.cursor = min (len(self.line), self.cursor+1)
        return True

    #________________________________________________________move_word_backward
    def move_word_backward (self):
        """ move cursor back one word """

        return True

    #_________________________________________________________move_word_forward
    def move_word_forward (self):
        """ move cursor forward one word """

        return True

    #______________________________________________________delete_char_backward
    def delete_char_backward (self):
        """ delete the character to the left of the cursor """

        if self.cursor:
            self.line = self.line[:self.cursor-1] + self.line[self.cursor:]
            self.cursor -= 1
        return True

    #_______________________________________________________delete_char_forward
    def delete_char_forward (self):
        """ delete the character to the right of the cursor """

        if self.cursor < len(self.line):
            self.line = self.line[:self.cursor] + self.line[self.cursor+1:]
        return True

    #______________________________________________________________kill_forward
    def kill_forward (self):
        """ kill the text from cursor position to the end of the line """

        if self.cursor < len(self.line):
            self.kill_buffer = self.line[self.cursor:]
            self.line = self.line[:self.cursor]        
        return True

    #_____________________________________________________________kill_backward
    def kill_backward (self):
        """ kill the text from cursor position to the start of the line """

        if self.cursor > 0:
            self.kill_buffer = self.line[:self.cursor]
            self.line = self.line[self.cursor:]
        return True

    #______________________________________________________________________yank
    def yank (self):
        """ yank the most recently killed text back at cursor position """

        cursor = self.cursor
        self.line = self.line[:cursor] + self.kill_buffer + self.line[cursor:]
        self.cursor += len(self.kill_buffer)
        if self.maxsize and len(self.line) > self.maxsize:
            self.line = self.line[:self.maxsize]
            self.cursor = min (len(self.line), self.cursor)
        return True

    #________________________________________________________________history_up
    def history_up (self):
        """ yank previous history line """

        if not len(self.history):
            return True
        if self.history_pos == len(self.history):
            self.copy = self.line
            self.replace (self.history[self.history_pos-1])
            self.history_pos -= 1
        elif self.history_pos > 0:
            self.replace (self.history[self.history_pos-1])
            self.history_pos -= 1
        return True

    #______________________________________________________________history_down
    def history_down (self):
        """ yank next history line """

        if not len(self.history):
            return True

        if self.history_pos == (len(self.history)-1):
            self.replace (self.copy)
            self.history_pos += 1
        elif self.history_pos < (len(self.history)-1):
            self.replace (self.history[self.history_pos+1])
            self.history_pos += 1
        return True

    #________________________________________________________write_history_file
    def write_history_file (self):
        """ save history to filename """

        file = open (self.filename, 'w')
        for l in self.history[-256:]:
            if len(l) > 0:
                file.write(l+'\n')
        file.close()

    #_________________________________________________________read_history_file
    def read_history_file (self):
        """ read history from filename """

        try:
            file = open (self.filename)
            self.history = []
            for l in file:
                self.history.append(l[:-1])
            file.close()
        except:
            pass

    #_____________________________________________________________clear_history
    def clear_history (self):
        """ clear history """

        pass


#__________________________________________________________________________main
if __name__ == '__main__':

    def completer(r, l):
        r.replace (l+' completed')
        
    print 'Performing tests'
    
    r = StringReadline(completer)

    r.start()                       # _
    for c in 'line 1': r.getc (c)   # line 1_
    assert (r.line == 'line 1')

    r.getc ('left')                 # line _1
    r.getc ('2')                    # line 2_1
    assert (r.line == 'line 21')
    
    r.getc ('backspace')            # line _1
    assert (r.line == 'line 1')

    r.getc ('delete')               # line _
    assert (r.line == 'line ')
    
    r.getc ('1')                    # line 1_
    r.getc ('2')                    # line 12_
    r.getc ('backspace')            # line 1_    
    assert (r.getc('\n') == True)   # line 1
    r.getc ('2')                    # line 1
    assert (r.getc('\n') == True)   # line 1

    r.start()                       # _
    for c in 'line 2': r.getc (c)   # line 2_
    assert (r.line == 'line 2')
    for i in range(20):
        r.getc ('left')             # _line 2
    for i in range(20):
        r.getc ('delete')           # _
    assert (r.line == '')
    r.replace ('line 2')            # line 2_
    assert (r.line == 'line 2')
    r.getc ('left')                 # line _2
    r.getc ('3')                    # line 3_2
    r.getc ('up')                   # line 1_
    assert (r.line == 'line 1')
    r.getc ('down')                 # line 32_
    assert (r.line == 'line 32')
    r.getc ('left')                 # line 3_2
    r.getc ('backspace')            # line _2
    assert (r.line == 'line 2')
    assert (r.getc('\n') == True)   # line 1

    r.start()                       # _
    for c in 'line 3': r.getc (c)   # line 3_
    r.getc ('control-a')            # _line 3
    r.getc ('control-k')            # _
    assert (r.line == '')    
    r.getc ('control-y')            # line 3_
    assert (r.line == 'line 3') 
    r.cancel()
    
    r.replace ('line 4')            # line 4_
    r.getc ('\t')                   # line 4 completed_
    assert (r.line == 'line 4 completed')
    assert (r.getc('\n') == True)   # line 4 completed
    
    r.start()                       # _
    for c in 'line 5': r.getc (c)   # line 5_
    try:
        r.getc ('control-c')        # line 5
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught !'
    assert (r.line == '')

    print "History: ", r.history
    print "Kill buffer: ", r.kill_buffer

