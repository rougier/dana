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
""" GTK Object attributes editor

    This package implements a Viewer class that is able to display and edit
    object attributes.
"""

import sys
import pygtk
pygtk.require("2.0")
import gtk, gtk.glade, pango


#____________________________________________________________________class View
class View (object):
    """ View class """
    
    def __init__ (self, name, object, attributes = [], predicate=None):
        
        self.name = name
        self.object = object
        self.attributes = attributes
        self.predicate = predicate
        
        # Build the list of attributes to be displayed
        if not self.attributes:
            attributes = []
            if self.predicate:
                for attr in dir(self.object):
                    if self.valid (attr):
                        self.attributes.append(attr)
            else:
                typenames = ['bool', 'int', 'long', 'float',
                             'str', 'unicode', 'list', 'tuple', 'dict']
                for attr in dir(self.object):
                    a = getattr(self.object,attr)
                    t = type(a).__name__
                    if t in typenames and not callable(a) and attr[0] != '_':
                        self.attributes.append(attr)



#__________________________________________________________________class Viewer
class Viewer(gtk.ScrolledWindow):
    """ Viewer class """

    _OBJECT_NAME    = 0
    _OBJECT         = 1
    _IS_OBJECT      = 2
    _VALUE_NAME     = 3
    _VALUE_REP      = 4
    _IS_VALUE       = 5
    _IS_EDITABLE    = 6
    _STYLE          = 7
    _WEIGHT         = 8

    #_________________________________________________________________ __init__
    def __init__(self, show_protected = True):
        """ """

        self.show_protected = show_protected

        # A view is a scrolled window
        gtk.ScrolledWindow.__init__(self)
        self.set_policy (gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        self.set_shadow_type(gtk.SHADOW_IN)
        
        # Tree model
        self.tree_model = gtk.TreeStore (
                        str, object, bool, str, str, bool, bool, int, int)
        self.root_model = self.tree_model.filter_new()
        self.root_model.set_visible_func(self.visibility)

        self.tree_view = gtk.TreeView (self.root_model)
        self.tree_view.set_rules_hint (True)
        self.tree_view.set_headers_visible (False)

        column = gtk.TreeViewColumn ("Attribute")
        column.set_sizing(gtk.TREE_VIEW_COLUMN_AUTOSIZE)
        renderer = gtk.CellRendererText ()
        column.pack_start (renderer,False)
        column.add_attribute(renderer, 'text', self._OBJECT_NAME)
        column.add_attribute(renderer, 'style', self._STYLE)
        column.add_attribute(renderer, 'visible', self._IS_OBJECT)
        column.add_attribute(renderer, 'weight', self._WEIGHT)
        
        renderer = gtk.CellRendererText ()
        column.pack_start (renderer)
        column.add_attribute(renderer, 'text', self._VALUE_NAME)
        column.add_attribute(renderer, 'style', self._STYLE)
        column.add_attribute(renderer, 'weight', self._WEIGHT)
        column.add_attribute(renderer, 'sensitive', self._IS_EDITABLE)
        self.tree_view.append_column (column)

        column = gtk.TreeViewColumn ("Value")
        column.set_expand(True)
        renderer = gtk.CellRendererText ()
        column.pack_start (renderer,False)
        renderer.connect ("edited", self.on_value_edited)
        column.add_attribute(renderer, 'text', self._VALUE_REP)
        column.add_attribute(renderer, 'visible', self._IS_VALUE)
        column.add_attribute(renderer, 'sensitive', self._IS_EDITABLE)
        column.add_attribute(renderer, 'editable', self._IS_EDITABLE)
        self.tree_view.append_column (column)

        self.add (self.tree_view)
        self.tree_view.get_selection().set_mode (gtk.SELECTION_SINGLE)
        self.tree_view.connect ('enter-notify-event', self.on_enter)
        self.tree_view.enable_model_drag_source (
                    gtk.gdk.BUTTON1_MASK,
                    [('text/plain', 0, 0)],
                    gtk.gdk.ACTION_DEFAULT| gtk.gdk.ACTION_COPY)
        self.tree_view.connect ("drag_data_get", self.drag_data_get_data)
        self.tree_view.connect('button-press-event',self.on_button_press)

    #________________________________________________________________visibility
    def visibility (self, model, iter, data=None):
        """ """
        if model.get_value (iter, self._IS_OBJECT):
            return True
        elif not model.get_value (iter, self._IS_EDITABLE):
            return self.show_protected
        return True

    #____________________________________________________________________append
    def append (self, view):
        """ """

        iter = None
        style = pango.STYLE_NORMAL
        weight = pango.WEIGHT_BOLD
        iter = self.tree_model.append (iter,
                  [view.name, view.object, True, '', '',
                   False, False, style, weight])
        
        # Attributes
        for attribute in view.attributes:
            readwrite = True
            style = pango.STYLE_NORMAL
            weight = pango.WEIGHT_BOLD
            try:
                setattr(view.object,attribute,getattr(view.object,attribute))
            except:
                readwrite = False

            name = attribute
            rep = repr (getattr (view.object, attribute))
            self.tree_model.append (iter,
                 [view.name, view.object, False, name, rep,
                  True, readwrite, style, weight])

#        self.tree_view.expand_row (self.tree_model.get_path(iter), True)
        
    #______________________________________________________________________sync
    def sync (self):
        """ """
        
        self.again = True
        while self.again:
            self.again = False
            self.tree_model.foreach (self.check_row)
        self.tree_model.foreach (self.update_row)

    #_________________________________________________________________check_row
    def check_row (self, model, path, iter):
        """ """
        
        if model.get_value(iter, self._IS_OBJECT):
            return False
        
        obj  = model.get_value (iter, self._OBJECT)
        name = model.get_value (iter, self._VALUE_NAME)
        if hasattr(obj, name):
            return False
        model.remove (iter)
        self.again = True
        return False

    #________________________________________________________________update_row
    def update_row (self, model, path, iter):
        """ """
        
        if model.get_value (iter, self._IS_OBJECT):
            return False

        obj = model.get_value (iter, self._OBJECT)
        name = model.get_value (iter, self._VALUE_NAME)
        rep = model.get_value (iter, self._VALUE_REP)
        if hasattr (obj, name):
            value = repr (getattr (obj, name))
            if value != rep:
                model.set_value (iter, self._VALUE_REP, value)

    #___________________________________________________________on_value_edited
    def on_value_edited (self, cell, path_string, new_text):
        """ """

        path = self.root_model.convert_path_to_child_path (path_string)
        iter = self.tree_model.get_iter (path)
        obj = self.tree_model.get_value (iter, self._OBJECT)
        name = self.tree_model.get_value (iter, self._VALUE_NAME)
        
        if hasattr (obj, name):
            a = getattr (obj, name)
            try:
                setattr (obj, name, eval(new_text))
            except:
                return
            rep = repr (getattr (obj, name))
            self.tree_model.set_value (iter, self._VALUE_REP, rep)

    #__________________________________________________________________on_enter
    def on_enter (self, event, widget):
        """ """

        self.sync()

    #________________________________________________________drag_data_get_data
    def drag_data_get_data (self,treeview,context,selection,target_id,etime):
        """ """

        treeselection = treeview.get_selection()
        model, iter = treeselection.get_selected()
        obj = model.get_value (iter, self._OBJECT_NAME)
        name = model.get_value (iter, self._VALUE_NAME)
        selection.set(selection.target, 8, obj + '.' + name)

    #___________________________________________________________on_button_press
    def on_button_press (self, widget, event):
        """  """

        if event.type == gtk.gdk.BUTTON_PRESS and event.button == 3:
            self.popup_menu(event)
            return True
        return False

    #________________________________________________________________popup_menu
    def popup_menu (self, event):
        """ Build a popup menu and make it appear """
        uidef = """
                <ui>
                    <popup name='PopupMenu'>
                        <menuitem name="hidden" action="Hidden" />
                    </popup>
                </ui>
                """
        ui = gtk.UIManager()
        actiongroup = gtk.ActionGroup('UIManager')
        actiongroup.add_toggle_actions(
            [('Hidden', None, '_Show all attributes','','Show all attributes',
                 self.toggle_show_protected,self.show_protected)])
        ui.insert_action_group(actiongroup,0)
        ui.add_ui_from_string(uidef)
        popup_menu = ui.get_widget('/PopupMenu')
        popup_menu.popup(None,None,None,event.button, event.time)

    #_____________________________________________________toggle_show_protected
    def toggle_show_protected (self, action):
        """  """
        
        self.show_protected = not self.show_protected
        self.root_model.refilter()


def Edit (view):
    """ """
    
    dialog = gtk.Dialog (title="Edit",
                         buttons = (gtk.STOCK_CLOSE, gtk.RESPONSE_CLOSE)
                          )
    dialog.set_has_separator(False)
    dialog.set_default_size(400, 400)
    dialog.set_border_width(8)
    dialog.vbox.set_spacing(6)
    viewer = Viewer ()
    viewer.append (view)
    dialog.vbox.pack_start (viewer, True, True)
    #    dialog.connect ('delete_event', lambda w,x: gtk.main_quit())
    dialog.show_all()


if __name__ == '__main__':
    dialog = gtk.Dialog (title="Edit",
                         buttons = (gtk.STOCK_CLOSE, gtk.RESPONSE_CLOSE)
                          )
    dialog.set_has_separator(False)
    dialog.set_default_size(400, 400)
    dialog.set_border_width(8)
    dialog.vbox.set_spacing(6)
    
    # We add some dummy attributes to the window instance
    dialog.a = 1
    dialog.b = 1.0
    dialog.c = True
    dialog.d = 'toto'
    dialog.e = [x for x in xrange(10)]
    
    
    # We use the gtk window itself to show its attribute
    viewer = Viewer ()
    viewer.append (View ('dialog1', dialog))
    viewer.append (View ('dialog', dialog))
    
    dialog.vbox.pack_start (viewer, True, True)
    dialog.connect ('delete_event', lambda w,x: gtk.main_quit())
    dialog.show_all()

    # We remove the b attribute and change a value to check synchronization
    del dialog.b
    dialog.c = False


    gtk.main()

    
