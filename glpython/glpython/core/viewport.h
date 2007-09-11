//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

// ============================================================================
//  Description:
// ----------------------------------------------------------------------------
//
//  A viewport represents a rectangular sub-area of the display and is meant to
//  be independent of other viewports. It possesses its own observer, list of
//  objects and if it has focus, it received and process events. Size and
//  position of the viewport can be in absolute (> 1) or relative coordinates
//  (< 1). 
//
//  Attributes
//    observer   - observer looking at the viewport
//    has_focus  - whether viewport has focus
//    has_border - whether viewport has surrounding border
//    size       - size request
//    position   - position request
//    geometry   - actual window relative geometry as a (x,y,w,h) tuple
//
// ============================================================================

#ifndef __GLPYTHON_CORE_VIEWPORT_H__
#define __GLPYTHON_CORE_VIEWPORT_H__

#include <vector>
#include "object.h"
#include "observer.h"

using namespace boost::python;


namespace glpython { namespace core {

    typedef boost::shared_ptr<class Viewport> ViewportPtr;
    
    class Viewport : public Object {
        public:
            ObserverPtr                 observer;
            bool                        has_focus;
            bool                        child_has_focus;
            bool                        has_border;
            float                       x,y,w,h;
            int                         _x,_y,_w,_h;
            int                         geometry[4];
            std::vector<ViewportPtr>    viewports;
            std::vector<ObjectPtr>      objects;

        public:
            Viewport (tuple size = make_tuple (1.0f, 1.0f),
                      tuple position = make_tuple(0.0f, 0.0f),
                      bool has_border = true,
                      bool is_ortho = false,
                      std::string name = "Viewport");
            virtual ~Viewport (void);

            virtual std::string repr (void);
            virtual void        append (ViewportPtr v);
            virtual void        append (ObjectPtr o);
            virtual int         len (void);
            virtual ObjectPtr   getitem (int index);
            virtual void        delitem (int index);
            virtual void        clear (void);

            virtual void   initialize (void);
            virtual void   render (void);
            virtual void   update (void);
            virtual void   select (int selection = 0);

            virtual void   set_is_ortho (bool is_ortho);
            virtual bool   get_is_ortho (void);
            virtual void   set_position (object position);
            virtual object get_position (void);
            virtual void   set_size (object size);
            virtual object get_size (void);
            virtual object get_geometry (void);
//          virtual void   set_geometry (object geometry);
            
            virtual void   select_event (int x, int y);
            virtual void   focus_event (int x, int y);
            virtual void   resize_event (int x, int y, int w, int h);
            virtual void   key_press_event (std::string key);
            virtual void   key_release_event (void);
            virtual void   button_press_event (int button, int x, int y);
            virtual void   button_release_event (int button, int x, int y);
            virtual void   pointer_motion_event (int x, int y);

            static void    python_export (void);
    };
}}

#endif
