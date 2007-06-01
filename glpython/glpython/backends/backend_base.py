#!/usr/bin/env python

#------------------------------------------------------------------------------
#
#   Copyright (c) 2007 Nicolas Rougier
# 
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
# 
#------------------------------------------------------------------------------
""" Abstraction of a window with an OpenGL context

This is an abstraction of a graphical window with an OpenGL context that
does not depend of a particular backend (such as wx, gtk, sdl, etc.) and
thus cannot be used directly, it must be fully specified for a given
backend by implementing event based functions.
"""

import time

class Window:
    """ Abstraction of a window with a double buffer RGBA OpenGL context"""
    
    def __init__(self, w=800, h=600, title='OpenGL window', fps=30.0):
        """ Window creation centered on screen. """
        
        self.fullscreened = False
        self.initialized = False
        if fps <= 0:
            fps = 50.0
        self.fps = fps
        self.delay = int(1000.0/float(self.fps))
        print self.delay
        self.external_event_loop = False
        self.title = title
        self.frames = 0
        self.t_start = 0
        self.actual_fps = self.fps

    def key_press (self, key):
        """ Key press accepts only one parameter describing the complete
        sequence of keys pressed. Special keys are named with symbolic names
        (for example escape, control, alt, f1, etc.) and any combination is
        made of key and '-' sign (for example, 'control-k').
        """
        pass
        
    def button_press (self, button, x, y):
        """ Button press accept 3 parameters describing the pressed button
        (from 1 to 5) and mouse position in window coordinates.
        """
        pass

    def button_release (self, x, y):
        """ Button release events accept 2 parameters describing mouse
        position in window coordinates.
        """
        pass

    def mouse_motion (self, x, y):
        """ Mouse motion events accept 2 parameters describing mouse position
        in window coordinates.
        """
        pass

    def resize (self, w=None, h=None):
        """ resize event
        """
        if w and h:
            self.width, self.height = w,h

    def init (self):
        """ initialization event
        """
        pass

    def render (self):
        """ render event
        """
        pass

    def title (self, title):
        """ Set window title. """
        pass

    def fullscreen (self):
        """ Enter fullscreen mode. """
        pass

    def unfullscreen (self):
        """ Leave fullscreen mode. """
        pass

    def show (self):
        """ Show window and run event loop. """
        pass

    def hide (self):
        """ Hide window """
        pass

    def destroy (self):
        """ Destroy window and quit event loop """
        pass

    def context_on (self):
        """ Setup OpenGL context for rendering """
        return True

    def context_off (self):
        """ Flush OpenGL context """
        return True

    def paint (self):
        """ Paint window """

        if not self.context_on ():
            return

        t = time.time()
        self.frames += 1
        if t - self.t_start > 5:
            self.actual_fps = self.frames / (t - self.t_start);
            self.t_start = t
            self.frames = 0
            self.set_title (self.title + ' %.1f fps' % self.actual_fps)

        if not self.initialized:
            self.init()
            self.initialized = True
        self.render ()
        self.context_off ()
        
    def idle (self):
        """ Process all waiting events (depend of backends implementation).
        """
        pass


if __name__ == '__main__':
    print
    print """This is a pure abstract class and """ \
          """cannot be run in stand-alone mode."""
    print
