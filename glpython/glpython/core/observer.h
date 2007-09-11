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
// An observer is a point in space describing the position of a virtual
// observer looking at point (0,0,0) using a specific camera and can be moved
// using mouse.
//
// Attributes:
//    phi:   rotation around x axis
//    theta: rotation aoound z axis (0 to pi)
// 
// ============================================================================

#ifndef __GLPYTHON_CORE_OBSERVER_H__
#define __GLPYTHON_CORE_OBSERVER_H__

#include "camera.h"


namespace glpython { namespace core {

    typedef boost::shared_ptr<class Observer> ObserverPtr;
    
    class Observer {
        public:
            std::string name;
            float       phi, theta;
            int         x,y, button;
            CameraPtr   camera;

        public:
            Observer (std::string name = "Observer");
            virtual ~Observer (void);

            virtual std::string repr (void);
            
            virtual void push (void);
            virtual void pop (void);
            
            virtual void resize_event (int x, int y, int w, int h);
            virtual void select_event (int x, int y);
            virtual void button_press_event (int button, int x, int y);
            virtual void button_release_event (int button, int x, int y);
            virtual void pointer_motion_event (int x, int y);

            static void python_export (void);
    };
}}

#endif
