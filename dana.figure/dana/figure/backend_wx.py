#!/usr/bin/env python
# ------------------------------------------------------------------------------
# DANA -- Distributed Asynchronous Numerical Adaptive computing library
# Copyright (c) 2007  Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------
""" Backend sub-class for the wxPython GUI environment

    This is one of the supported backend types (gtk being the other one).
"""

import wx
from wx import glcanvas
import OpenGL.GL as GL
import backend

class Backend (backend.Backend):
    """ Backend sub-class for the wxPython GUI environment """

    def __init__ (self, figure, fps=30.0):
        """ figure -- figure to be rendered
            fps    -- frame per second
        """

        backend.Backend.__init__ (self, figure, fps)
        self.size = 0,0
        self.app = wx.App (False)
        style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE
        self.frame = wx.Frame  (None, -1, 'DANA Figure',
                                wx.DefaultPosition,
                                wx.Size (int(figure.size[0]*72),
                                         int(figure.size[1]*72)),
                                style)
        self.frame.SetClientSize (
            wx.Size (int(figure.size[0]*72), int(figure.size[1]*72)))
        #self.frame.CenterOnScreen ()
        attribList = (wx.glcanvas.WX_GL_RGBA,
                      wx.glcanvas.WX_GL_DOUBLEBUFFER,
                      wx.glcanvas.WX_GL_DEPTH_SIZE, 16)
        self.glcanvas = glcanvas.GLCanvas (self.frame, attribList=attribList)
        self.frame.Bind    (wx.EVT_CLOSE,            self._on_close)
        self.glcanvas.Bind (wx.EVT_ERASE_BACKGROUND, self._on_erase_background)
        self.glcanvas.Bind (wx.EVT_KEY_DOWN,         self._on_key_press)
        self.glcanvas.Bind (wx.EVT_SIZE,             self._on_size)
        self.glcanvas.Bind (wx.EVT_LEFT_DOWN,        self._on_button_press)
        self.glcanvas.Bind (wx.EVT_MIDDLE_DOWN,      self._on_button_press)
        self.glcanvas.Bind (wx.EVT_LEFT_UP,          self._on_button_release)
        self.glcanvas.Bind (wx.EVT_MIDDLE_UP,        self._on_button_release)
        self.glcanvas.Bind (wx.EVT_MOTION,           self._on_mouse_motion)
        self.glcanvas.Bind (wx.EVT_MOUSEWHEEL,       self._on_mouse_wheel)
        self.glcanvas.Bind (wx.EVT_PAINT,            self._on_paint)
        self._drag_in_process = False
        self._timer = None


    def _on_erase_background (self, event):
        """ Hacks to avoid flicker on MSWin """

        pass


    def _on_paint (self, event):
        """ Actual paint of the scene """
        
        dc = wx.PaintDC (self.glcanvas)
        self.glcanvas.SetCurrent()
        if not self._initialized:
            self.initialize ()
        self.render()
        GL.glFlush()
        self.glcanvas.SwapBuffers()


    def _on_size (self, event):
        """ Resize event """

        self.size = self.glcanvas.GetClientSize()
        if self.glcanvas.GetContext():
            GL.glViewport (0,0,self.size[0],self.size[1])
            self.setup()
            self.glcanvas.Refresh()
        event.Skip()

    def _on_mouse_wheel (self, event):
        """ Mouse scroll buttons (zoom in or out) """

        x,y = event.GetX(), event.GetY()
        if event.GetWheelRotation() > 0:
            self.zoom_in ()
        elif event.GetWheelRotation() < 0:
            self.zoom_out ()
        self.glcanvas.Refresh()

    def _on_key_press (self, event):
        """ Keyboard events """

        if event.GetKeyCode() == wx.WXK_ESCAPE:
            if not self.figure.reset():
                w, h = self.figure.size
                width,height = self.glcanvas.GetClientSize()
                s = max (width, height)
                self.frame.SetClientSize (
                    wx.Size (int(self.figure.normalized_size[0]*s),
                             int(self.figure.normalized_size[1]*s)))
            self.glcanvas.Refresh()
        elif event.GetKeyCode() == wx.WXK_SPACE:
            self.figure.save ('figure.pdf')
            print "Figure saved in 'figure.pdf'"
        elif (event.GetRawKeyCode() > 32 and event.GetRawKeyCode() < 128):
            if event.ControlDown() and chr(event.GetRawKeyCode()) in ['Q', 'q']:
                self.hide()
            elif event.ControlDown() and chr(event.GetRawKeyCode()) in ['W', 'w']:
                self.hide()


    def _on_button_press (self, event):
        """ Mouse button press """

        x,y = event.GetX(), event.GetY()
        w,h = self.glcanvas.GetClientSizeTuple()
        if event.LeftIsDown():
            z = float(max (w,h))
            self.drag_start ( (x/z, y/z) )
            self.glcanvas.Refresh()
        elif event.MiddleIsDown():
            self.select (x,y)

    def _on_button_release (self, event):
        """ Mouse button releae """
        self.drag_end ()


    def _on_mouse_motion (self, event):
        """ Mouse motion """

        if not self._drag_in_process:
            return
        x,y = event.GetX(), event.GetY()
        w,h = self.glcanvas.GetClientSize()
        z = float(max (w,h))
        self.drag_to ( (x/z, y/z) )
        self.glcanvas.Refresh()


    def _on_timeout (self):
        """ Timeout callback """
        
        self.glcanvas.Refresh ()
        if self._delay and self._timer:
            self._timer.Restart (self._delay)


    def _on_close (self, event):
        """ Close event """

        self.hide()

    def show (self):
        """ Show the backend on screen and strat event loop """

        self.frame.Show (True)
        self.glcanvas.SetFocus()
        self._mainloop = False
        if self._delay:
            self._timer = wx.CallLater (self._delay, self._on_timeout)
        if not wx.App.IsMainLoopRunning():
            self.app.MainLoop()
            self._mainloop = True

    def hide (self):
        """ Hide backend """

        if self._timer:
            self._timer.Stop()
        self.frame.Hide ()
        if not self._mainloop:
            self.app.Exit()
