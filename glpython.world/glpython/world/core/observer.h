//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
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
// This Observer inherits glpython::core::Observer.
// The only difference with the mother class is the push method
//
// Attributes:
//    phi:   rotation around x axis
//    theta: rotation aoound z axis (0 to pi)
// 
// ============================================================================

#ifndef __GLPYTHON_WORLD_CORE_OBSERVER_H__
#define __GLPYTHON_WORLD_CORE_OBSERVER_H__

#include "glpython/core/observer.h"
#include "util.h"
#include <GL/glu.h>

namespace glpython { namespace world { namespace core {

    class Observer : public glpython::core::Observer {
    public:
        float position[3];
        float look_at[3];
        float forward[3];
        float left[3];
        float up[3];
        
        float theta,phi;

        bool allow_movement;

    public:
        Observer (std::string name = "Camera Roger");
        virtual ~Observer (void);

        virtual void push (void);
        virtual void pop(void);

        virtual void render (void);
        void local_axes(void);
        
        // Input Control commands
        void rotate(float dtheta, float dphi);
        void center();
        void VectorsFromAngles(void);

        virtual void pointer_motion_event (float x, float y);

        static void python_export (void);
    };
}}}


#endif
