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

""" Interactive OpenGL ipython terminal

This implements an interactive ipython session in an OpenGL window.

Shortcuts:
----------
    Ctrl-A : goto line start
    Ctrl-E : goto line end
    Ctrl-K : clear line from cursor to end
    Ctrl-L : clear console
    Ctrl-S : save session
    Tab:     completion
"""

import re, sys, os
from StringIO import StringIO
try:
    import IPython
except Exception, e:
    raise "Error importing IPython (%s)" % str(e)    
import rlcompleter
from console import *
from history import *
from io import *
from font import _FontMarker, _ColorMarker

ansi_colors =  {'0;30': Black,
                '0'   : Black,
                '0;31': Red,
                '0;32': Green,
                '0;33': Brown,
                '0;34': Blue,
                '0;35': Purple,
                '0;36': Cyan,
                '0;37': LightGray,
                '1;30': DarkGray,
                '1;31': DarkRed,
                '1;32': SeaGreen,
                '1;33': Yellow,
                '1;34': LightBlue,
                '1;35': MediumPurple,
                '1;36': LightCyan,
                '1;37': White}


class IterableIPShell:
    """ Iterable IPython shell
    
    Adapted from Accerciser (http://live.gnome.org/Accerciser)
    @author: Eitan Isaacson
    @organization: IBM Corporation
    @copyright: Copyright (c) 2007 IBM Corporation
    @license: BSD

    All rights reserved. This program and the accompanying materials are made 
    available under the terms of the BSD which accompanies this distribution,
    and is available at U{http://www.opensource.org/licenses/bsd-license.php}
    """
    
    def __init__(self,argv=None,user_ns=None,user_global_ns=None,
                 cin=None, cout=None,cerr=None, input_func=None):
        """ """
        
        if input_func:
            IPython.iplib.raw_input_original = input_func
        if cin:
            IPython.Shell.Term.cin = cin
        if cout:
            IPython.Shell.Term.cout = cout
        if cerr:
            IPython.Shell.Term.cerr = cerr
        if argv is None:
            argv=[]

        # This is to get rid of the blockage that occurs during 
        # IPython.Shell.InteractiveShell.user_setup()
        IPython.iplib.raw_input = lambda x: None

        self.term = IPython.genutils.IOTerm(cin=cin, cout=cout, cerr=cerr)
        os.environ['TERM'] = 'dumb'
        excepthook = sys.excepthook
        self.IP = IPython.Shell.make_IPython(
                            argv,
                            user_ns=user_ns,
                            user_global_ns=user_global_ns,
                            embedded=True,
                            shell_class=IPython.Shell.InteractiveShell)
        self.IP.system = lambda cmd: self.shell(
                                self.IP.var_expand(cmd),
                                header='IPython system call: ',
                                verbose=self.IP.rc.system_verbose)
        sys.excepthook = excepthook
        self.iter_more = 0
        self.history_level = 0
        self.complete_sep =  re.compile('[\s\{\}\[\]\(\)]')

    def open (self, filename):
        """ Open and execute a given filename """

#        self.write ("execfile('%s')" % filename)
#        self.execute()

    def execute(self):
        """ """
        self.history_level = 0
        orig_stdout = sys.stdout
        sys.stdout = IPython.Shell.Term.cout
        try:
            line = self.IP.raw_input(None, self.iter_more)
            if self.IP.autoindent:
                self.IP.readline_startup_hook(None)
        except KeyboardInterrupt:
            self.IP.write('\nKeyboardInterrupt\n')
            self.IP.resetbuffer()
            # keep cache in sync with the prompt counter:
            self.IP.outputcache.prompt_count -= 1
            if self.IP.autoindent:
                self.IP.indent_current_nsp = 0
            self.iter_more = 0
        except:
            self.IP.showtraceback()
        else:
            self.iter_more = self.IP.push(line)
            if (self.IP.SyntaxTB.last_syntax_error and self.IP.rc.autoedit_syntax):
                self.IP.edit_syntax_error()
        if self.iter_more:
            self.prompt = str(self.IP.outputcache.prompt2).strip()
            if self.IP.autoindent:
                self.IP.readline_startup_hook(self.IP.pre_readline)
        else:
            self.prompt = str(self.IP.outputcache.prompt1).strip()
        sys.stdout = orig_stdout

    def updateNamespace(self, ns_dict):
        """ """
        self.IP.user_ns.update(ns_dict)


    def complete(self, line):
        """ """
        split_line = self.complete_sep.split(line)
        possibilities = self.IP.complete(split_line[-1])
        if possibilities:
            common_prefix = reduce(self._commonPrefix, possibilities)
            completed = line[:-len(split_line[-1])]+common_prefix
        else:
            completed = line
        return completed, possibilities

    def _commonPrefix(self, str1, str2):
        """ """
        for i in range(len(str1)):
            if not str2.startswith(str1[:i+1]):
                return str1[:i]
        return str1

    def shell(self, cmd,verbose=0,debug=0,header=''):
        """ """
        stat = 0
        if verbose or debug: print header+cmd
        # flush stdout so we don't mangle python's buffering
        if not debug:
            input, output = os.popen4(cmd)
        print output.read()
        output.close()
        input.close()



class Terminal (Console, IterableIPShell):
    """ Interactive OpenGL console class """

    def __init__(self, namespace={}, paint=None, idle=None):
        """ Initialize console """
        
        namespace['terminal'] = self
        self.cout = StringIO()
        self.line = ''
        Console.__init__(self)
        IterableIPShell.__init__(self, user_global_ns = namespace,
                                 cout=self.cout, cerr=self.cout,
                                 input_func=self.raw_input)
        self.stdin = Infile (self, sys.stdin.fileno())
        self.sys_stdin = sys.stdin
        self.interrupt = False                                 
        self.execute()
        self.paint = paint
        self.idle = idle
        self.input = ''
        self.input_mode = False
        self.color_pat = re.compile('\x01?\x1b\[(.*?)m\x02?')
        self.history = History()
        self.banner()


    def raw_input(self, prompt=''):
        """ """
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        if self.line != '':
            return self.line
        return self.current_line()

    def banner(self):
        """ Display mixed Python/IPython banner """

        Console.write (self,
           'Python %s on %s\n' % (sys.version, sys.platform),
           font=Bold)
        Console.write (self,
           'IPython %s -- An enhanced Interactive Python.\n\n' % IPython.__version__,
           font=Bold)
                       
        self.show_prompt(self.prompt)


    def show_prompt (self, prompt):
        """ Display prompt """
        
        self.write (prompt)
        self.protect = len (self.buffer[-1])-1


    def write (self, text):
        """ """
        t = ""
        segments = self.color_pat.split(text)
        segment = segments.pop(0)
        Console.write (self,segment)
        if segments:
            ansi_tags = self.color_pat.findall(text)
            for tag in ansi_tags:
                i = segments.index(tag)
                t += ansi_colors[tag] +segments[i+1]
                segments.pop(i)
        Console.write (self, t)

    
    def showReturned (self,text):
        self.write (text)
        if text:
            Console.write (self,'\n')
        self.show_prompt (self.prompt)


    def _processLine(self):
        """ """
        self.history_pos = 0
        sys.stdin = self.stdin
        self.execute()
        sys.stdin  = self.sys_stdin
        rv = self.cout.getvalue()
        if rv: rv = rv.strip('\n')
        self.showReturned(rv)
        self.cout.truncate(0)


    def key_press (self, key):
        """ Key pressed handler """
        
        if key == 'enter':
            if self.input_mode:
                self.input_mode = False
                self.input = self.current_line()
                self.write('\n')
            else:
                self.line = self.current_line()
                self.history.append (self.line) 
                self.move_end()
                self.write('\n')
                self._processLine()
                self.line = ''
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
            self.write('\n')
            self.interrupt = True
            self.input_mode = False
            self._processLine()
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
                if c == ' ':
                    tabs += c
                else:
                    break
            
            # Find possible completions and longest common prefix
            stripped_line = line.strip(" \t")
            if not stripped_line:
                return
            completions = []
            self.completer = rlcompleter.Completer(globals())
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
                self.show_prompt(self.prompt)
                self.write (tabs+prefix)
                self.linestart = iter
                return

            i = 0
            while i<len(completions):
                for j in range(item_per_line):
                    self.write (completions[i].ljust(max_completion_len)+' ')
                    i += 1
                    if i >= len(completions):
                        break
                self.write ('\n')
            self.write ('\n')
            self.allow_prompt = True
            self.show_prompt(self.prompt)
            self.write (tabs + prefix)
        elif len(key) == 1 and key >= ' ' and key <= '~':
            Console.write(self,key)


