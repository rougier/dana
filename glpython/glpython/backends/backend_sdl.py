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
""" SDL window with an OpenGL context """

import os, time, threading
import pygame, pygame.key
from pygame.locals import *
import backend_base as base


class Window (base.Window):
    """ SDL window with an OpenGL context """
    
    def __init__(self, w=800, h=600, title='SDL OpenGL window', fps=30):
        """ Window creation at given size, centered on screen. """
        
        base.Window.__init__(self, w, h, title, fps)
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.init()
        pygame.key.set_repeat (500, 50)
        pygame.display.set_caption(title)
        self.window = pygame.display.set_mode((w,h), OPENGL|DOUBLEBUF)
        self.fullscreened = False
        self.timer = False
        self.width, self.height = w,h


    def key_press_event (self, event):
        """ Key press event """

        keymap = {pygame.K_BACKSPACE : "backspace", pygame.K_TAB : "tab",
                  pygame.K_RETURN : "enter",        pygame.K_ESCAPE : "escape",
                  pygame.K_DELETE : "delete",       pygame.K_END : "end",
                  pygame.K_HOME : "home",           pygame.K_LEFT : "left",
                  pygame.K_UP : "up",               pygame.K_RIGHT : "right",
                  pygame.K_DOWN : "down",           pygame.K_F1 : "f1",
                  pygame.K_F2 : "f2",               pygame.K_F3 : "f3",
                  pygame.K_F4 : "f4",               pygame.K_F5 : "f5",
                  pygame.K_F6 : "f6",               pygame.K_F7 : "f7",
                  pygame.K_F8 : "f8",               pygame.K_F9 : "f9",
                  pygame.K_F10 : "f10",             pygame.K_F11 : "f11",
                  pygame.K_F12 : "f12",             pygame.K_KP_ENTER : "enter"}

        key = ''
        keycode = event.key
        keyname = keymap.get (keycode, None)
        mods = pygame.key.get_mods()
        
        if keyname:
            key = keyname
            self.key_press (key)
        elif mods & pygame.KMOD_CTRL and keycode >= 32 and keycode <=127:
            key = 'control-%c' % chr(keycode)
            self.key_press (key)
        elif mods & pygame.KMOD_ALT and keycode >= 32 and keycode <=127:
            key = 'alt-%c' % chr(keycode)
            self.key_press (key)
        else:
            key = str(event.unicode)
            if key and ord(key) >= 32 and ord(key) <= 127:
                self.key_press (key)
        self.paint()


    def button_press_event (self, event):
        """ Button press event """

        button = event.button
        self.button_press (button, event.pos[0], event.pos[1])
        self.paint()


    def button_release_event (self, event):
        """ Button release event """

        self.button_release (event.pos[0], event.pos[1])
        self.paint()


    def mouse_motion_event (self, event):
        """ Mouse motion event """

        self.mouse_motion (event.pos[0], event.pos[1])
        self.paint()

    def resize_event (self, (w,h)):
        """ Window resize event """
        
        if h == 0:
            h = 1
        self.resize (w, h)


    def title (self, title):
        """ Set window title """

        pygame.display.set_caption (title)


    def fullscreen (self):
        """ Fullscreen mode """
        
        if not self.fullscreened:
            pygame.display.toggle_fullscreen()
            self.fullscreened = True


    def unfullscreen (self):
        """ Windowed mode """

        if self.fullscreened:
            pygame.display.toggle_fullscreen()
            self.fullscreened = False        


    def context_off (self):
        """ Flush OpenGL context """

        pygame.display.flip()


    def idle (self):
        """ Idle function """

        while 1:
            event = pygame.event.poll()
            if event.type == NOEVENT:
                break
            elif event.type == KEYDOWN:
                self.key_press_event (event)
            elif event.type == QUIT:
                break


    def show (self):
        """ Show window and run main loop """

        base.Window.show(self)
        self.resize_event ((self.width, self.height))
        self.done = False
        
        t = time.time()
        first_pass = True
        while not self.done:
            while 1:
                event = pygame.event.poll()
                if event.type == NOEVENT:
                    break
                elif event.type == KEYDOWN:
                    self.key_press_event (event)
                elif event.type == MOUSEBUTTONDOWN:
                    self.button_press_event (event)
                elif event.type == MOUSEBUTTONUP:
                    self.button_release_event (event)
                elif event.type == MOUSEMOTION:
                    self.mouse_motion_event (event)
                elif event.type == QUIT:
                    self.done = True

            elapsed = time.time() - t
            if elapsed > self.delay/1000.0 or first_pass:
                self.paint()
                first_pass = False
            t = time.time()


    def hide (self):
        """ Hide window """
        
        pass
        
        
    def destroy (self):
        """ Destroy window """
        
        self.done = True        



if __name__ == '__main__':
    from OpenGL.GL import *
    
    def resize(self,w=None,h=None):
        glViewport (0,0,w,h)

    def init(self):
        glClearColor (0.0, 0.0, 0.0, 0.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)

    def render(self):
        glClear (GL_COLOR_BUFFER_BIT)
        glColor3f (1.0, 1.0, 1.0)
        glBegin(GL_POLYGON)
        glVertex3f (0.25, 0.25, 0.0)
        glVertex3f (0.75, 0.25, 0.0)
        glVertex3f (0.75, 0.75, 0.0)
        glVertex3f (0.25, 0.75, 0.0)
        glEnd()

    window = Window (512,256, fps = 1.0)
    Window.init = init
    Window.render = render
    Window.resize = resize
    window.debug = True
    window.show ()
