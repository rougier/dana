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
""" GTK window with an OpenGL context """

import sys
import pygtk, gobject
import gtk.gtkgl
import backend_base as base


class Window (base.Window):
    """ GTK window with an OpenGL context """
    
    def __init__(self, w=800, h=600, title='GTK OpenGL window', fps=30):
        """ Window creation centered on screen. """
        
        base.Window.__init__(self, w, h, title, fps)
        self.window = gtk.Window()
        self.window.set_position (gtk.WIN_POS_CENTER)
        self.window.set_title (title)
        self.window.set_reallocate_redraws (True)
        self.window.connect ('delete-event', self.delete_event)
        attrib_list = (gtk.gdkgl.DOUBLEBUFFER,
                       gtk.gdkgl.RGBA,
                       gtk.gdkgl.DEPTH_SIZE, 1)
        glconfig = gtk.gdkgl.Config (attrib_list)
        self.glarea = gtk.gtkgl.DrawingArea (glconfig)
        self.glarea.connect ('configure_event', self.resize_event)
        self.glarea.connect ('motion_notify_event', self.mouse_motion_event)
        self.glarea.connect ('scroll_event', self.scroll_event)        
        self.glarea.connect ('button_press_event', self.button_press_event)
        self.glarea.connect ('button_release_event', self.button_release_event)
        self.window.connect ('key_press_event', self.key_press_event)
        self.window.set_default_size (w,h)
        self.glarea.set_events (gtk.gdk.EXPOSURE_MASK
                                | gtk.gdk.BUTTON_PRESS_MASK
                                | gtk.gdk.BUTTON_RELEASE_MASK                               
                                | gtk.gdk.POINTER_MOTION_MASK
                                | gtk.gdk.POINTER_MOTION_HINT_MASK)
        self.window.add (self.glarea)
        self.width, self.height = w,h
        self.timeout_id = 0
        self.window.realize()


    def key_press_event (self, widget, event):
        """ Key press event """

        key = ''
        if event.state & gtk.gdk.CONTROL_MASK:
            key = 'control-'
        if event.state & gtk.gdk.MOD1_MASK:
            key = 'alt-'
        if event.keyval == gtk.keysyms.Return:
            key += 'enter'
        elif event.keyval in (gtk.keysyms.KP_Up, gtk.keysyms.Up):
            key += 'up'
        elif event.keyval in (gtk.keysyms.KP_Down, gtk.keysyms.Down):
            key += 'down'
        elif event.keyval in (gtk.keysyms.KP_Left, gtk.keysyms.Left):
            key += 'left'
        elif event.keyval in (gtk.keysyms.KP_Right, gtk.keysyms.Right):
            key += 'right'
        elif event.keyval >= gtk.keysyms.F1 and event.keyval <= gtk.keysyms.F12:
            key += 'f%d' % (event.keyval-gtk.keysyms.F1+1)
        elif event.keyval == gtk.keysyms.Escape:
            key += 'escape'
        elif event.keyval == gtk.keysyms.BackSpace:
            key += 'backspace'
        elif event.keyval == gtk.keysyms.Delete:
            key += 'delete'
        elif event.keyval == gtk.keysyms.Home:
            key += 'home'
        elif event.keyval == gtk.keysyms.End:
            key += 'end'
        elif event.keyval in (gtk.keysyms.KP_Tab, gtk.keysyms.Tab):
            key += 'tab'
        elif event.keyval >= 32 and event.keyval <= 127:
            key += chr(event.keyval)
        self.key_press (key)
        self.paint()

    
    def delete_event (self, widget, event):
        """ Delete event """
        
        if self.external_event_loop:
            self.window.hide()
        else:
            gtk.main_quit()
        return True


    def button_press_event (self, widget, event):
        """ Button press event """

        self.button_press (event.button, event.x, event.y)
        self.paint()

    def button_release_event (self, widget, event):
        """ Button release event """

        self.button_release (event.x, event.y)
        self.paint()

    def scroll_event (self, widget, event):
        """ Scroll event """

        if event.direction == gtk.gdk.SCROLL_UP:
            self.button_press (4, event.x, event.y)
        elif event.direction == gtk.gdk.SCROLL_DOWN:
            self.button_press  (5, event.x, event.y)
        self.paint()

    def mouse_motion_event (self, widget, event):
        """ Mouse motion event """

        if event.is_hint:
            x, y, state = event.window.get_pointer()
        else:
            x = event.x
            y = event.y
            state = event.state
        self.mouse_motion (x, y)
        self.paint()

    def resize_event (self, widget=None, event=None):
        """ Window resize event """
        
        glcontext = self.glarea.get_gl_context()
        gldrawable = self.glarea.get_gl_drawable()
        if not gldrawable.gl_begin(glcontext):
            return
        x, y, w, h = self.glarea.get_allocation()
        self.resize (w, h)
        gldrawable.gl_end()
        self.paint()
        return True


    def timeout_event (self):
        """ Timeout function """
        
        self.paint ()
        return True


    def title (self, title):    
        """ Set window title """

        self.window.set_title (title)
        
        
    def fullscreen (self):
        """ Enter fullscreen mode """
        
        self.window.fullscreen()
        

    def unfullscreen (self):
        """ Leave fullscreen mode """
        
        self.window.unfullscreen()


    def context_on (self):
        """ Switch on OpenGL context for rendering """

        glcontext = self.glarea.get_gl_context()
        gldrawable = self.glarea.get_gl_drawable()
        if not gldrawable.gl_begin(glcontext):
            return False
        return True


    def context_off (self):
        """ Switch off OpenGL context """

        gldrawable = self.glarea.get_gl_drawable()
        if gldrawable.is_double_buffered():
            gldrawable.swap_buffers()
        else:
            glFlush()
        gldrawable.gl_end()


    def idle (self):
        """ Idle function """

        while gtk.events_pending():
            gtk.main_iteration()
        

    def show (self):
        """ Show window and run event loop """
        
        if self.timeout_id:
            gobject.source_remove (self.timeout_id)
        self.timeout_id = gobject.timeout_add (self.delay, self.timeout_event)
        self.window.show_all()
        self.resize_event ()
        
        if self.external_event_loop:
            return
        base.Window.show(self)
        if not gtk.main_level():
            gtk.main()
            self.external_event_loop = False
        else:
            self.external_event_loop = True


    def hide (self):
        """ Hide window """

        if self.timeout_id:
            gobject.source_remove (self.timeout_id)
            self.timeout_id = 0
        base.Window.hide(self)
        self.window.hide()


    def destroy (self):
        """ Destroy window """

        self.hide()
        base.Window.destroy (self)
        if not self.external_event_loop:
            gtk.main_quit()
            return


if __name__ == '__main__':
    from OpenGL.GL import *
    
    def resize(self, w=None,h=None):
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
