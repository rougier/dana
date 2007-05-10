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

""" Interactive OpenGL python terminal

This implements an interactive python session in an OpenGL window.

Shortcuts:
----------
    Ctrl-A : goto line start
    Ctrl-E : goto line end
    Ctrl-K : clear line from cursor to end
    Ctrl-L : clear console
    Ctrl-S : save session
    Tab:     completion
"""

import sys, os.path
import rlcompleter
import traceback
from console import *
from history import *
from io import *

if not hasattr(sys, 'ps1'): sys.ps1 = '>>> '
if not hasattr(sys, 'ps2'): sys.ps2 = '... '



class Terminal (Console):
    """ Interactive OpenGL console class """

    def __init__(self, namespace={}, paint=None, idle=None):
        """ Initialize console """

        Console.__init__(self)
        self.paint = paint
        self.idle = idle
        self.input_mode = False
        
        # Setup completer
        namespace['terminal'] = self        
        self.completer = rlcompleter.Completer(namespace)

        # Internal setup
        self.namespace = namespace
        self.cmd = ''
        self.input = ''
        self.input_mode = False
        self.linestart = 0
        self.session = []
        self.session_filename = "session.txt"
        self.allow_prompt = True
        
        # Setup hooks for standard output.
        self.stdout = Outfile (self, sys.stdout.fileno(), self.write_stdout)
        self.stderr = Outfile (self, sys.stderr.fileno(), self.write_stderr)
        self.stdin  = Infile (self, sys.stdin.fileno())

        # Save system standards I/O
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        self.sys_stdin  = sys.stdin

        # Setup command history
        self.history = History()
        self.namespace['__history__'] = self.history
        self.banner()

    def banner(self):
        """ Display python banner """

        self.write ('Python %s on %s\n' % (sys.version, sys.platform), font=Bold)
        self.write ('''Type "help", "copyright", "credits" '''
                    '''or "license" for more information.\n\n''', font=Bold)
        self.prompt1()


    def prompt (self):
        """ Display prompt """
        
        if not self.allow_prompt:
            return
        Console.write (self, self.promptline)
        self.protect = len (self.buffer[-1])-1
        self.allow_prompt = False


    def prompt1 (self):
        """ Display normal prompt """
        
        self.promptline = sys.ps1
        self.prompt()


    def prompt2 (self):
        """ Display continuation prompt """
        
        self.promptline = sys.ps2
        self.prompt()

    
    def write_stdout (self, text):
        """ Write text in blue """
        
        Console.write (self, text, Blue)
        self.protect = len (self.buffer[-1])
        self.cursor = -1


    def write_stderr (self, text):
        """ Write text in red """
            
        Console.write (self, text, Red)
        self.protect = len (self.buffer[-1])
        self.cursor = -1


    def is_balanced (self, line):
        """ Checks line balance for brace, bracket, parenthese and string quote

        This helper function checks for the balance of brace, bracket,
        parenthese and string quote. Any unbalanced line means to wait until
        some other lines are fed to the console.
        """
        
        s = line
        s = filter(lambda x: x in '()[]{}"\'', s)
        s = s.replace ("'''", "'")
        s = s.replace ('"""', '"')
        instring = False
        brackets = {'(':')', '[':']', '{':'}', '"':'"', '\'':'\''}
        stack = []
        
        while len(s):
            if not instring:
                if s[0] in ')]}':
                    if stack and brackets[stack[-1]] == s[0]:
                        del stack[-1]
                    else:
                        return False
                elif s[0] in '"\'':
                    if stack and brackets[stack[-1]] == s[0]:
                        del stack[-1]
                        instring = False
                    else:
                        stack.append(s[0])
                        instring = True
                else:
                    stack.append(s[0])
            else:
                if s[0] in '"\'' and stack and brackets[stack[-1]] == s[0]:
                    del stack[-1]
                    instring = False
            s = s[1:]
        return len(stack) == 0


    def eval (self):
        """ Evaluate if current line is ready for execution """

        l = self.current_line()
        self.cursor = -1
        self.write ('\n')
        self.history.append (l) 
        self.paint()

        if l == '':
            cmd = self.cmd
            self.cmd = ''
            self.execute (cmd)
            self.prompt1()
            return

        self.cmd = self.cmd + l + '\n'
        if not self.is_balanced (self.cmd):
            self.prompt2()
            return
        l = l.rstrip()
        if len(l) > 0:
            if l[-1] == ':' or l[-1] == '\\' or l[0] in ' \11':
                self.prompt2()
                return

        cmd = self.cmd
        self.cmd = ''
        self.execute (cmd)
        self.prompt1()
        return


    def execute (self, cmd):
        """ Execute a given command """

        self.session.append (cmd)
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        sys.stdin  = self.stdin
        try:
            try:
                r = eval (cmd, self.namespace, self.namespace)
                if r is not None:
                    print `r`
            except SyntaxError:
                exec cmd in self.namespace
        except:
            if hasattr (sys, 'last_type') and sys.last_type == SystemExit:
                self.quit_handler()
            else:
                try:
                    tb = sys.exc_traceback
                    if tb:
                        tb=tb.tb_next
                    traceback.print_exception (sys.exc_type, sys.exc_value, tb)
                except:
                    sys.stderr, self.stderr = self.stderr, sys.stderr
                    traceback.print_exc()
        if sys.stdout is not self.stdout:
            sys.stdout = self.sys_stdout
            sys.stderr = self.sys_stderr
            sys.stdin  = self.sys_stdin


    def open (self, filename):
        """ Open and execute a given filename """

        self.write ("Executing '%s'...\n" % filename, 'script')
        self.execute ("execfile('%s')" % filename)
        self.prompt1()


    def key_press (self, key):
        """ Key pressed handler """
        
        # Enter
        if key == 'enter':
            if self.input_mode:
                self.input_mode = False
                self.input = self.current_line()
                self.write('\n')
            else:
                self.allow_prompt = True
                self.eval()
        
        # Previous command
        elif key == 'up':
            if not self.input_mode:
                self.replace (self.history.prev (self.current_line()))
        elif key == 'down':
            if not self.input_mode:
                self.replace (self.history.next (self.current_line()))
        elif key == 'left':
            self.move (-1)
        elif key == 'right':
            self.move (+1)
        elif key == 'backspace':
            self.delete (self.cursor, self.cursor-1)
        elif key == 'delete':
            if self.cursor < -1:
                self.delete (self.cursor, self.cursor+1)
                self.move (+1)
        elif key == 'end' or key == 'control-e':
            self.move_end()
        elif key == 'home' or key == 'control-a':
            self.move_start()
        elif key == 'control-c':
            self.interrupt = True
            self.input_mode = False
        elif key == 'control-l':
            self.clear()
        elif key == 'control-k':
            if self.cursor < -1:
                self.delete (self.cursor, -1)
                self.move_end()
        elif key == 'tab':
            # If line empty, it's a real tab
            line = self.current_line()
            if not line.strip():
                self.write ('    ')
                return False
            tabs = ''
            for c in line:
                if c == ' ': tabs += c
                else:          break
            
            # Find possible completions and longest common prefix
            stripped_line = line.strip(" \t")
            if not stripped_line:
                return
            completions = []
            completion = self.completer.complete (stripped_line,0)
            
            # No completion
            if not completion:
                return

            max_completion_len = len(completion)
            prefix = completion
            j = 1
            while completion:
                max_completion_len = max (len(completion),max_completion_len)
                if completion not in completions:
                    completions.append (completion)
                    i = 0
                    while (i < min(len(prefix),len(completion))
                           and prefix[0:i] == completion[0:i]):
                        i += 1
                    if  prefix[0:i] != completion[0:i]:
                        prefix = prefix[0:i-1]
                completion = self.completer.complete (stripped_line,j)
                j += 1

            # One completion only  
            if len(completions) == 1:
                try:
                    if callable(eval(completions[0])):
                        completions[0] += "()"
                except:
                    pass
                self.replace (tabs+completions[0])
                return

            # Several possible completions        
            item_per_line = self.columns  / (max_completion_len+1)
            self.write ('\n')
            if len(completions) > 256:
                self.write (
                    "More than 256 possibilities (%d)\n\n" % len(completions),
                    'completion')
                self.allow_prompt = True
                self.prompt()
                self.write (tabs+prefix)
                self.linestart = iter
                return

            i = 0
            while i<len(completions):
                for j in range(item_per_line):
                    self.write_stdout (
                     completions[i].ljust(max_completion_len)+' ')
                    i += 1
                    if i >= len(completions):
                        break
                self.write ('\n')
            self.write ('\n')
            self.allow_prompt = True
            self.prompt()
            self.write (tabs + prefix)
        elif len(key) == 1 and key >= ' ' and key <= '~':
            self.write(key)


