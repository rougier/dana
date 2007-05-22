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
""" Python shell emulation

"""

import sys, os.path
import rlcompleter
import traceback
from io import *

if not hasattr(sys, 'ps1'): sys.ps1 = '>>> '
if not hasattr(sys, 'ps2'): sys.ps2 = '... '


#___________________________________________________________________class Shell
class Shell:

    #_________________________________________________________________ __init__
    def __init__(self, terminal, namespace = locals()):
        """ Initialization """

        self.namespace = namespace
        self.terminal = terminal
        self.cmd = ''
        self.banner()

    #_____________________________________________________________________write
    def write (self, text):
        """ write text """
        self.terminal.write (text)

    #____________________________________________________________________banner
    def banner(self):
        """ banner display """

        self.write ('Python %s on %s\n' % (sys.version, sys.platform))
        self.write ('''Type "help", "copyright", "credits" '''
                    '''or "license" for more information.\n \n''')
        self.prompt1()

    #____________________________________________________________________prompt
    def prompt (self):
        """ prompt line """

        self.terminal.scroll = 0
        self.terminal.read (sys.ps1, self.completer)

    #___________________________________________________________________prompt1
    def prompt1 (self):
        """ prompt line 1 """

        self.terminal.scroll = 0
        self.terminal.read (sys.ps1, self.completer)

    #___________________________________________________________________prompt2
    def prompt2 (self):
        """ prompt line 2 """

        self.terminal.scroll = 0
        self.terminal.read (sys.ps2, self.completer)

    #_______________________________________________________________is_balanced
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

    #______________________________________________________________________eval
    def eval (self, line):
        """ Evaluate if line is ready for execution """

        if line == '':
            cmd = self.cmd
            self.cmd = ''
            self.execute (cmd)
            self.prompt1()
            return
        self.cmd = self.cmd + line + '\n'
        if not self.is_balanced (self.cmd):
            self.prompt2()
            return
        line = line.rstrip()
        if len(line) > 0:
            if line[-1] == ':' or line[-1] == '\\' or line[0] in ' \11':
                self.prompt2()
                return
        cmd = self.cmd
        self.cmd = ''
        self.execute (cmd)
        self.prompt1()

    #___________________________________________________________________execute
    def execute (self, cmd):
        """ Execute a given command """

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
                        tb = tb.tb_next
                    traceback.print_exception (sys.exc_type, sys.exc_value, tb)
                except:
                    sys.stderr, self.stderr = self.stderr, sys.stderr
                    traceback.print_exc()

    #_________________________________________________________________completer
    def completer (self, rl, line):
        """ Complete current line """

        # Line's empty, we insert a real tab (4 spaces indeed)
        if not line.strip():
            for i in range(4):
                rl.getc(' ')

        # Get current tabulation at line start
        tabs = ''
        for c in line:
            if c == ' ':
                tabs += c
            else:
                break
        
        # Find possible completions and longest common prefix
        stripped_line = line.strip(" \t")
        if not stripped_line:
            return
        completions = []
        _completer = rlcompleter.Completer (self.namespace)
        completion = _completer.complete (stripped_line,0)
        
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
            completion = _completer.complete (stripped_line,j)
            j += 1

        # One completion only  
        if len(completions) == 1:
            try:
                if callable(eval(completions[0])):
                    completions[0] += "()"
            except:
                pass
            rl.replace (tabs+completions[0])
            return

        # Several possible completions        
        item_per_line = self.terminal.columns  / (max_completion_len+1)
        text = "\n"
        if len(completions) > 256:
            text += "More than 256 possibilities (%d)\n\n" % len(completions)
            self.write (text)
            return

        i = 0
        while i<len(completions):
            for j in range(item_per_line):
                text += completions[i].ljust(max_completion_len)+' '
                i += 1
                if i >= len(completions):
                    break
            text += '\n'
        text += '\n'
        self.write (text)


