#! /bin/env python
#
import sys, os, glob
import string
from HTMLParser import HTMLParser

path = 'manual/'

class Parser (HTMLParser):
    title = None
    parent = None
    next = None
    prev = None
    contents = None
    last_starttag = None
    last_endtag = None

    def __init__ (self):
        """
        """
        HTMLParser.__init__ (self)
        self.div_stack = 0
        self.nav_stack = -1
        self.in_title = False
        self.in_h1 = False
        self.in_navigation = 0

    def handle_starttag (self, tag, attrs):
        """
        """
        
        self.last_starttag = tag
        self.last_endtag = None
        

        if tag == 'h1':
            self.in_h1 = True

        if tag == 'div':
            self.div_stack += 1

        if tag == 'div' and attrs == [('class', 'navigation')]:
            self.nav_stack = self.div_stack-1
            self.in_navigation = 2

        elif tag == 'title':
            self.in_title = True
            
        elif tag == 'a' and ('rel', 'prev') in attrs:
            for a,v in attrs:
                if a == 'href':
                    self.prev  = v
                elif a == 'title':
                    self.prev_title = v
        elif tag == 'a' and ('rel', 'next') in attrs:
            for a,v in attrs:
                if a == 'href':
                    self.next = v
                elif a == 'title':
                    self.next_title = v
        elif tag == 'a' and ('rel', 'contents') in attrs:
            for a,v in attrs:
                if a == 'href':
                    self.contents = v
                elif a == 'title':
                    self.contents_title = v

        elif tag == 'a' and ('rel', 'parent') in attrs:
            for a,v in attrs:
                if a == 'href':
                    self.parent = v
                elif a == 'title':
                    self.parent_title = v

    def handle_data (self,data):
        """
        """
        
        if self.in_title:
            self.title = data
        elif self.in_h1:
            pass

    def handle_endtag (self, tag):
        """
        """

        self.last_endtag = tag

        if tag == 'title':
            self.in_title = False
        elif tag == 'h1':
            self.in_h1 = False
        
        if tag == 'div':
            self.div_stack -= 1
            if self.nav_stack == self.div_stack:
                self.in_navigation = 1
                self.nav_stack = -1

    


for filename in glob.glob( os.path.join(path, '*.html') ):
#for filename in glob.glob( os.path.join(path, 'node4.html') ):
    print "Processing %s" % filename

    parser = Parser()

#    f = open (filename)
#    for line in f:
#        parser.feed (line)
#    f.close()

    lines = []
    f = open (filename)
    for line in f:
        parser.feed (line)
        if parser.last_starttag == 'body':
            lines.append ('<body>\n')
            lines.append ('<div class="page">\n')
#            lines.append ('<h1>%s</h1>\n' % parser.title)
#            lines.append ('<table width="100%" class="header"><tr>\n')
#            lines.append ('<td class="left" width="25%"><a href="index.html">DANA</a></td>\n')
#            lines.append ('<td class="right" width="75%%">%s</td>\n' % parser.title)
#            lines.append ('</tr></table>\n')
            lines.append ('<div class="content">\n')            
        elif parser.last_endtag == 'body':
            lines.append ('</div>\n')
            lines.append ('<table width="100%" class="footer"><tr>\n')
            if parser.prev:
                lines.append ('<td class="left" width="50%%"><a href="%s">%s</a></td>\n' % (parser.prev, parser.prev_title))
            else:
                lines.append ('<td class="left" width="50%"></td>\n')
            if parser.next:
                lines.append ('<td class="right" width="50%%"><a href="%s">%s</a></td>\n' % (parser.next, parser.next_title))
            else:
                lines.append ('<td class="right" width="50%"></td>\n')       
            lines.append ('</tr></table>\n')
            lines.append ('</div>\n')
            lines.append ('</body>\n')
        elif parser.in_h1:
            pass
        elif line == "</h1>\n":
            lines.append ('<h1>%s</h1>\n' % parser.title)
            if hasattr(parser, 'parent_title'):
                lines.append ('<div class="nav_right"><a href="%s">%s</a></div>\n' % (parser.parent, parser.parent_title))
        elif parser.in_navigation == 0:
            lines.append (line)
        elif parser.in_navigation == 1:
            parser.in_navigation = 0
    f.close()

#<div class="titlepage">
#<p>
#<span style='font-size: 350%;'><strong>D.A.N.A.</strong><br/></span>
#<span style='font-size: 125%;'> Distributed Asynchronous Numeric and Adaptive computing<br/></span>
#<span style='text-align: right; font-size:80%'> Copyright (c) 2007 Nicolas Rougier</span>
#<div class='center'>
#<p>Release 1.0, August 13, 2007</p>
#</div>


    f = open(filename, "w")
    for line in lines:
        f.write(line)
    f.close()

#    print "   -> Title:", parser.title    
#    print "   -> Prev:", parser.prev
#    print "   -> Next:", parser.next
#    print "   -> Parent:", parser.parent
#    print "   -> Contents:", parser.contents
#    print
    
    parser.close()

    
