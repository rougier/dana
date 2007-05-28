#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------

import os.path
import pygtk
pygtk.require("2.0")
import gobject, gtk, gtk.glade
from dana.gui.data import datadir


#____________________________________________________________class ControlPanel
class ControlPanel:
    """ ControlPanel class"""

    #_________________________________________________________________ __init__
    def __init__ (self, model):
        """ Initialization """
        
        # Some checkings on provided model
        if not hasattr(model, 'stop') or not callable (model.stop):
            print "Provided model does not have a 'stop' function"
            return

        if not hasattr(model, 'start') or not callable (model.start):
            print "Provided model does not have a 'start' function"
            return

        if not hasattr(model, 'age') or callable (model.age):
            print "Provided model does not have 'age' attribute"
            return

        if not hasattr(model, 'running') or callable (model.running):
            print "Provided model does not have 'running' attribute"
            return

        xmlfile = os.path.join (datadir(), 'control_panel.glade')

        # Actual initialization
        self.model = model
        self.source_id = 0
        self.start_time = model.age
        self.end_time = 0
        self.tree = gtk.glade.XML (xmlfile)
        self.tree.signal_autoconnect (self)
        self.dialog.connect('destroy', self.on_destroy)
        self.dialog.connect('delete-event', self.on_delete)
        self.dialog.show()
        
        if model.running:
            self.button_start.clicked()

    #______________________________________________________________ __getattr__
    def __getattr__(self, name):
        """ Convenient method to access widgets as normal attributes """

        try:
            return self.tree.get_widget (name)
        except:
            raise AttributeError, name

    #________________________________________________________________on_destroy
    def on_destroy (self, widget):
        """ Destroy event """

        return True

    #_________________________________________________________________on_delete
    def on_delete (self, widget, event):
        """ Delete event """

        return True

    #_____________________________________________________________on_use_epochs
    def on_use_epochs (self, widget):
        """ """

        b = self.toggle_use_epochs.get_active()
        self.spin_epochs.set_sensitive (b)

    #___________________________________________________________________on_step
    def on_step (self, widget):
        """ """
        self.start(1)  

    #__________________________________________________________________on_start
    def on_start (self, widget):
        """ """

        epochs = 0
        if self.toggle_use_epochs.get_active():
            epochs = self.spin_epochs.get_value()
        self.start (epochs)

    #___________________________________________________________________on_stop
    def on_pause (self, widget):
        """ """
        pass

    #___________________________________________________________________on_stop
    def on_stop (self, widget):
        """ """

        self.stop()

    #________________________________________________________________on_timeout
    def on_timeout (self):
        """ """
        
        if self.epochs:
            fraction = ( (self.model.age - self.start_time) /
                         float (self.end_time - self.start_time) )
            self.progressbar.set_fraction (fraction)
            self.progressbar.set_text ('%d/%d' %
              (self.model.age-self.start_time,self.end_time - self.start_time))
        else:
            self.progressbar.pulse()
            self.progressbar.set_text (
                        '%d' % (self.model.age-self.start_time))

        if not self.model.running:
            self.progressbar.set_text (' ')
            self.source_id = 0
            self.on_stop(self.button_stop)
            return False
        return True

    #________________________________________________________________on_timeout
    def start (self, epochs=0):
        """ """
    
        epochs = int (epochs)

        self.toggle_use_epochs.set_sensitive (False)
        self.button_step.set_sensitive (False)
        self.button_start.set_sensitive (False)
        self.spin_epochs.set_sensitive (False)
        self.epochs = epochs        
        if epochs:
            self.start_time = self.model.age
            self.end_time = self.model.age+self.epochs
        else:
            self.start_time = self.model.age
            self.end_time = 0
        if self.source_id:
            gobject.source_remove (self.source_id)
            self.source_id = 0
        self.source_id = gobject.timeout_add (100, self.on_timeout)
        
        self.progressbar.set_fraction (0)
        self.progressbar.set_text (' ')
        if self.epochs:
            self.model.start(self.epochs)
        else:
            self.model.start()

    #________________________________________________________________on_timeout
    def stop (self):
        """ """

        self.model.stop()
        self.toggle_use_epochs.set_sensitive (True)
        self.button_step.set_sensitive (True)
        self.button_start.set_sensitive (True)
        self.spin_epochs.set_sensitive (True)
        self.progressbar.set_fraction (0)
        self.progressbar.set_text (' ')
        if self.source_id:
            gobject.source_remove (self.source_id)
            self.source_id = 0

#__________________________________________________________________________main
if __name__ == '__main__':
    import time, thread, sys

    class Model:
        def __init__ (self):
            self.age = 0
            self.running = False

        def start (self, epochs=0):
            if not self.running:
                self.running = True
                thread.start_new_thread (self.iterate, (epochs,) )
                return  True
            else:
                return  False

        def iterate (self, epochs):
            self.do_stop = False
            if epochs:
                while epochs > 0 and not self.do_stop:
                    time.sleep(.01)
                    epochs -= 1
                    self.age += 1
            else:
                while not self.do_stop:
                    time.sleep(.01)
                    self.age += 1
            self.running = False

        def stop (self):
            self.do_stop = True
            self.running = False

    model = Model()
    panel = ControlPanel(model)
    gtk.gdk.threads_init()
    gtk.main()

