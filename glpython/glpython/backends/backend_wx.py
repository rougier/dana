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
""" WX backend """

import wx
from wx import glcanvas
import backend_base as base


class Window (base.Window):
    """ WX backend """
    
    def __init__(self, w=800, h=600, title='WX OpenGL window', fps=30):
        """ Window creation at given size, centered on screen. """
        
        base.Window.__init__ (self, w, h, title, fps)
        self.app = wx.App(False)
        self.frame = wx.Frame (None,-1,title, wx.DefaultPosition,wx.Size(w, h))
        self.frame.CenterOnScreen()
        self.canvas = glcanvas.GLCanvas (self.frame)
        self.frame.Bind  (wx.EVT_CLOSE, self.close_event)
        self.canvas.Bind (wx.EVT_KEY_DOWN, self.key_press_event)
        self.canvas.Bind (wx.EVT_SIZE, self.resize_event)
        self.canvas.Bind (wx.EVT_LEFT_DOWN, self.button_press_event)
        self.canvas.Bind (wx.EVT_MIDDLE_DOWN, self.button_press_event)
        self.canvas.Bind (wx.EVT_RIGHT_DOWN, self.button_press_event)
        self.canvas.Bind (wx.EVT_LEFT_UP, self.button_release_event)
        self.canvas.Bind (wx.EVT_MIDDLE_UP, self.button_release_event)
        self.canvas.Bind (wx.EVT_RIGHT_UP, self.button_release_event)
        self.canvas.Bind (wx.EVT_MOTION, self.mouse_motion_event)
        self.canvas.Bind (wx.EVT_MOUSEWHEEL, self.mouse_wheel_event)
        self.width, self.height = w,h


    def close_event (self, event):
        """ Close event """
        
        if event.CanVeto() and self.external_event_loop:
            event.Veto()
            self.hide()
        else:
            self.hide()
            self.app.ExitMainLoop()


    def key_press_event (self, event):
        """ Key press event """

        keymap = {wx.WXK_BACK : "backspace",      wx.WXK_TAB : "tab",
                  wx.WXK_RETURN : "enter",        wx.WXK_ESCAPE : "escape",
                  wx.WXK_DELETE : "delete",       wx.WXK_END : "end",
                  wx.WXK_HOME : "home",           wx.WXK_LEFT : "left",
                  wx.WXK_UP : "up",               wx.WXK_RIGHT : "right",
                  wx.WXK_DOWN : "down",           wx.WXK_F1 : "f1",
                  wx.WXK_F2 : "f2",               wx.WXK_F3 : "f3",
                  wx.WXK_F4 : "f4",               wx.WXK_F5 : "f5",
                  wx.WXK_F6 : "f6",               wx.WXK_F7 : "f7",
                  wx.WXK_F8 : "f8",               wx.WXK_F9 : "f9",
                  wx.WXK_F10 : "f10",             wx.WXK_F11 : "f11",
                  wx.WXK_F12 : "f12",             wx.WXK_NUMPAD_TAB : "tab",
                  wx.WXK_NUMPAD_ENTER : "enter",  wx.WXK_NUMPAD_HOME : "home",
                  wx.WXK_NUMPAD_LEFT : "left",    wx.WXK_NUMPAD_UP : "up",
                  wx.WXK_NUMPAD_RIGHT : "right",  wx.WXK_NUMPAD_DOWN : "down",
                  wx.WXK_NUMPAD_END : "end",  wx.WXK_NUMPAD_DELETE : "delete"}
        key = ''
        keycode = event.GetRawKeyCode()
        keyname = keymap.get (event.GetKeyCode(), None)
        if event.ControlDown():
            key = 'control-'
        if event.AltDown():
            key += 'alt-'
        
        if keyname:
            key += keyname
            self.key_press (key)
        elif keycode >= 32 and keycode <= 127:
            key += chr(keycode)
            self.key_press (key)
        self.paint()

    def mouse_wheel_event (self, event):
        """ Mouse wheel event """
        
        x,y = event.GetX(), event.GetY()
        if event.GetWheelRotation() > 0:
            self.button_press (4, x, y)
        elif event.GetWheelRotation() < 0:
            self.button_press (5, x, y)
        self.paint()



    def button_press_event (self, event):
        """ Mouse button event """

        x,y = event.GetX(), event.GetY()
        button = 0
        if event.LeftIsDown():
            button = 1
        elif event.MiddleIsDown():
            button = 2
        elif event.RightIsDown():
            button = 3
        self.button_press (button, x, self.height-y)
        self.paint()


    def button_release_event (self, event):
        """ Mouse button event """

        x,y = event.GetX(), event.GetY()
        self.button_release (x, self.height-y)
        self.paint()


    def mouse_motion_event (self, event):
        """ Mouse motion event """

        x,y = event.GetX(), event.GetY()
        self.mouse_motion (x, self.height-y)
        self.paint()


    def resize_event (self, event=None):
        """ Window resize event """
        
        size = self.canvas.GetClientSize()
        if self.canvas.GetContext():
            self.canvas.SetCurrent()
            self.resize (size.width, size.height)
            self.paint()

    def timeout_event (self):
        """ Timeout function """
        
        self.paint ()
        self.timer.Restart (self.delay)


    def title (self, title):    
        """ Set window title """

        self.frame.SetTitle (title)


    def fullscreen (self):
        """ Fullscreen mode """
        
        self.frame.ShowFullScreen (True)
        

    def unfullscreen (self):
        """ Windowed mode """
        
        self.frame.ShowFullScreen (False)


    def context_on (self):
        """ Setup OpenGL context for rendering """
        
        dc = wx.PaintDC (self.canvas)
        self.canvas.SetCurrent()
        return True


    def context_off (self):
        """ Flush OpenGL context """
        
        self.canvas.SwapBuffers()
        

    def idle (self):
        """ Idle function """

        self.app.Yield()


    def show (self):
        """ Show window and run main loop """

        self.frame.Show ()
        self.canvas.SetFocus()
        self.timer = wx.CallLater (self.delay, self.timeout_event)
        self.resize_event ()
        
        if self.external_event_loop:
            return
        base.Window.show (self)
        if not wx.App.IsMainLoopRunning():
            self.app.MainLoop()
            self.external_event_loop = False
        else:
            self.external_event_loop = True


    def hide (self):
        """ Hide window """

        self.timer.Stop()
        base.Window.hide(self)
        self.frame.Hide ()


    def destroy (self):
        """ Destroy window """

        base.Window.destroy (self)
        self.frame.Destroy()
        if not self.external_event_loop:
            self.app.ExitMainLoop()



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

    window = Window (512,256, fps=1.0)
    Window.init = init
    Window.render = render
    Window.resize = resize
    window.debug = True
    window.show ()
