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
from StringIO import StringIO
import IPython
from io import *
    
if not hasattr(sys, 'ps1'): sys.ps1 = '>>> '
if not hasattr(sys, 'ps2'): sys.ps2 = '... '



#___________________________________________________________________class Shell
class Shell:

    #_________________________________________________________________ __init__
    def __init__(self, terminal, namespace = locals()):
        """ Intialization """

        self.namespace = namespace
        self.terminal = terminal
        self.cmd = ''
        self.cout = self.terminal.stdout #StringIO()
        self.cerr = self.terminal.stderr #StringIO()
        IPython.Shell.Term.cout = self.cout
        IPython.Shell.Term.cerr = self.cerr
        argv=[]
        IPython.iplib.raw_input = lambda x: None
        excepthook = sys.excepthook
        self.IP = IPython.Shell.make_IPython(
            argv, user_ns=namespace, user_global_ns=namespace,
            embedded=True, shell_class=IPython.Shell.InteractiveShell)

        self.IP.system = lambda cmd: self.shell (
            self.IP.var_expand(cmd),
            header='IPython system call: ',
            verbose=self.IP.rc.system_verbose)
        sys.excepthook = excepthook
        self.banner()

    #_____________________________________________________________________shell
    def shell (self, cmd, verbose=False, debug=False, header=''):
        """ Shell commands redirection """

        # TODO: find how to tell shell the size of our terminal
        if verbose or debug:
            print header+cmd
        if not debug:
            pipe_input, pipe_output = os.popen4(cmd)
        self.write_stdout (pipe_output.read())
        pipe_output.close()
        pipe_input.close()

    #_____________________________________________________________________write
    def write (self, text):
        """ Write text """
        self.terminal.write (text)

    #______________________________________________________________write_stdout
    def write_stdout (self, line):
        """ write on stdout """
        self.terminal.write_stdout (line)

    #______________________________________________________________write_stderr
    def write_stderr (self, line):
        """ write on stderr """
        self.terminal.write_stderr (line)

    #____________________________________________________________________banner
    def banner(self):
        self.write ('Python %s on %s\n' % (sys.version, sys.platform))
        self.write ('''Type "help", "copyright", "credits" '''
                    '''or "license" for more information.\n''')
        self.write ('''IPython %s -- An enhanced Interactive Python.\n \n'''
                        % IPython.__version__)
        self.prompt1()

    #____________________________________________________________________prompt
    def prompt (self):
        self.terminal.scroll = 0
        self.terminal.read (sys.ps1, self.completer)

    #___________________________________________________________________prompt1
    def prompt1 (self):
        self.terminal.scroll = 0
        self.terminal.read (sys.ps1, self.completer)

    #___________________________________________________________________prompt2
    def prompt2 (self):
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

        line = self.IP.multiline_prefilter (line, '')
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

        self.IP.push (cmd)
        if (self.IP.SyntaxTB.last_syntax_error and self.IP.rc.autoedit_syntax):
            self.IP.edit_syntax_error()

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
        
        self.terminal.write_input()
        text = "\n"
        if len(completions) > 256:
            text += "More than 256 possibilities (%d)\n" % len(completions)
            self.terminal.write (text, self.terminal.output_buffer)
            return

        i = 0
        while i<len(completions):
            for j in range(item_per_line):
                text += completions[i].ljust(max_completion_len)+' '
                i += 1
                if i >= len(completions):
                    break
            text += '\n'
        self.terminal.write (text, self.terminal.output_buffer)


