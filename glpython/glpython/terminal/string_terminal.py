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
""" Pseudo ansi string terminal 

    This class implements a pseudo-ansi terminal within a string buffer. User
    has to manually handle input via the getc function  and display actual
    output/input buffers when it is necessary.
"""

import re
from string_readline import StringReadline

#__________________________________________________________class StringTerminal
class StringTerminal:
    """ StringTerminal class """

    #_________________________________________________________________ __init__
    def __init__ (self):
        """ Initialization """
        
        self.columns = 80
        self.lines = 24
        self.cursor = (0,0)
        self.output_buffer = []
        self.input_buffer = []
        self.rl = StringReadline ()
        self.read_status = False
        self.ansi_re = re.compile('\x01?\x1b\[(.*?)m\x02?')
        self.codes = {'0':  'normal',              '1':  'bold',
                      '2':  'faint',               '3':  'standout',
                      '4':  'underline',           '5':  'blink',
                      '7':  'reverse',             '8':  'invisible',
                      '00': 'normal',              '01': 'bold',
                      '02': 'faint',               '03': 'standout',
                      '04': 'underline',           '05': 'blink',
                      '07': 'reverse',             '08': 'invisible',
                      '22': 'normal',              '23': 'no-standout',
                      '24': 'no-underline',        '25': 'no-blink',
		              '27': 'no-reverse',          '30': 'black foreground',
                      '31': 'red foreground',      '32': 'green foreground',
                      '33': 'yellow foreground',   '34': 'blue foreground',
                      '35': 'magenta foreground',  '36': 'cyan foreground',
                      '37': 'white foreground',    '39': 'default foreground',
                      '40': 'black background',    '41': 'red background',
                      '42': 'green background',    '43': 'yellow background',
                      '44': 'blue background',     '45': 'magenta background',
                      '46': 'cyan background',     '47': 'white background',
                      '49': 'default background'}

    #_____________________________________________________________________clear
    def clear (self):
        """ Clear terminal """
        
        output_buffer = []

    #______________________________________________________________________read
    def read (self, prompt = '> ', completer=None):
        """ input start """

        self.prompt = ''
        if self.output_buffer:
            for m, l in self.output_buffer[-1]:
                self.prompt += l
            del self.output_buffer[-1]
        self.prompt += prompt

            
        self.rl.completer = completer
        self.rl.start()
        self.read_status = True
        self.input_buffer = []
        self.write (self.prompt, self.input_buffer)
        self.prompt_len = 0
        if self.input_buffer:
            for markup,line in self.input_buffer[-1]:
                self.prompt_len += len(line)
        c = self.rl.cursor + self.prompt_len                
        self.cursor = (c%self.columns, c/self.columns)

    #______________________________________________________________________getc
    def getc (self, c):
        """ Feed c to the readline function """

        try:
            result = self.rl.getc(c)
        except KeyboardInterrupt:
            self.write('\n', self.output_buffer)
            self.write('\nKeyboardInterrupt\n')
            self.rl.line = ''
            self.read_status = False
            raise KeyboardInterrupt
        if result and self.read_status:
            self.write ('\n' + self.prompt+self.rl.line, self.output_buffer)
            c = self.rl.cursor + self.prompt_len                
            self.cursor = (c%self.columns, c/self.columns)
            self.input_line = self.rl.line
            self.input_buffer = []
            self.read_status = False
        elif not result:
            self.input_buffer = []
            self.write (self.prompt+self.rl.line, self.input_buffer)
            c = self.rl.cursor + self.prompt_len                
            self.cursor = (c%self.columns, c/self.columns)
        return result

    #_______________________________________________________________write_input
    def write_input (self):
        """ Write current input on output """

        if len(self.output_buffer) and self.output_buffer[-1][-1][1] != '':
            self.write ('\n', self.output_buffer)
        self.write (self.prompt + self.rl.line, self.output_buffer)


    #_____________________________________________________________________write
    def write (self, line, buffer=None):
        """ write line at current position """

        if self.read_status and buffer == None:
            self.write (self.prompt + self.rl.line, self.output_buffer)

        segments = self.ansi_re.split(line)
        markup_style = []
        markup_bg = ''
        markup_fg = ''
        segment = segments.pop(0)
        self._add (segment, [], buffer)
        
        if segments:
            tags = self.ansi_re.findall(line)
            for tag in tags:
                i = segments.index(tag)
                codes = tag.split(';')
                for code in codes:
                    if code in self.codes.keys():
                        m = self.codes[code]
                        if 'foreground' in m:
                            markup_fg = m
                            if m == 'default foreground':
                                markup_fg = None
                        elif 'background' in m:
                            markup_bg = m
                            if m == 'default background':
                                markup_bg = None
                        elif m == 'normal':
                            markup_style = ['normal']
                            markup_fg = None
                            markup_bg = None
                        elif m[:3] == 'no-' and m[3:] in markup_style:
                            markup_style.remove (m[3:])
                        elif m not in markup_style:
                            markup_style.append (m)
                markup = markup_style
                if markup_fg: markup.append (markup_fg)
                if markup_bg: markup.append (markup_bg)
                self._add (segments[i+1], markup, buffer)
                if 'normal' in markup_style:
                    markup_style = []
                segments.pop(i)

    #_______________________________________________________________________add
    def _add (self, text, markup=[], buffer=None):
        """ add text with given markup at end of buffer """
    
        if not text:
            return

        if buffer == None:
            buffer = self.output_buffer

        # Compute last line size and last markup
        p = 0
        last_markup = None
        if len(buffer):
            for m, l in buffer[-1]:
                last_markup = m
                p += len(l)

        # Check if we have foreground or background colors in markup
        foreground_set = True
        background_set = True
        for m in markup:
            if 'foreground' in m:
                foreground_set = True
            elif 'background' in m:
                backround_set = True

        # Mix last line markup and text markup
        if 'normal' in markup:
            markup = []
        else:
            if last_markup:
                for m in last_markup:
                    if 'foreground' in m and not foreground_set:
                        markup.append (m)
                        foreground_set = True
                    elif 'background' in m and not background_set:
                        markup.append (m)
                        bakground_set = True
                for m in ['bold', 'faint', 'invisible']:
                    if m in last_markup and m not in markup:
                        markup.append (m)
            for m in ['standout', 'underline', 'blink', 'reverse']:
                if 'no-'+m in markup:
                    markup.remove ('no-'+m)
                elif last_markup and m in last_markup:
                    markup.append (m)
        markup.sort()
        
        # Insert new text with markup
        if text[0] == '\n':
            p = 0
        long_lines = text.split ('\n')
        lines = []
        for line in long_lines:
            while len(line) > (self.columns-p):
                lines.append (line[0:self.columns-p])
                line = line[self.columns-p:]
                p = 0
            lines.append (line)
        if len(buffer):
            if buffer[-1][-1][0] == markup:
                buffer[-1][-1][1] += lines[0]
            elif buffer[-1][-1][0] == [] and buffer[-1][-1][1] == '':
                buffer[-1][-1] = [markup, lines[0]]
            else:
                buffer[-1].append ([markup, lines[0]])
            del lines[0]
        for line in lines:
            buffer.append ( [[markup, line], ] )
            if buffer != self.input_buffer:
                self.last_line = line


#__________________________________________________________________________main
if __name__ == '__main__':

    t = StringTerminal()
    def completer(rl, line):
        t.write ('\ncompletion\n')

    t.read (">>> ")
    for c in 'hello': t.getc(c)
    t.getc('\n'), t.write('\n')

    t.write('Hello\n')
    t.write('\033[1mee\n')
    t.read ('>')
    

    print "OUTPUT"
    for i in t.output_buffer:
        print i

    print "INPUT"
    for i in t.input_buffer:
        print i


