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
//   A robot
// 
// ============================================================================

#ifndef __GLPYTHON_WORLD_CORE_ROBOT_H__
#define __GLPYTHON_WORLD_CORE_ROBOT_H__

#include "glpython/core/object.h"
#include "glpython/core/viewport.h"
#include "util.h"
#include <GL/glu.h>	
#include <GL/gl.h>

namespace glpython { namespace world { namespace core {

    typedef boost::shared_ptr<class Robot> RobotPtr;
    
    class Robot : public glpython::core::Object {

    public:
        glpython::core::ViewportPtr view; // Viewport for the camera

        float h_camera;

        float position[3]; // Absolute position in space
        float look_at[3]; // Absolute position in space

        float theta,phi;

        float forward[3]; // Direction
        float left[3];   // Direction
        float up[3];    // Direction
        
        // Constructor , Destructor
        Robot (std::string name = "Robot");
        virtual ~Robot (void);

        // Render methods
        virtual void render (void);
        virtual void robot (float dx, float dy, float dz); // height,width,depth
        void local_axes(void);
        
        // Input Control commands
        void move(float dx, float dy, float dz); // dx : forward/backward
                                                 // dy : left/right
                                                 // dz : up/down
        void rotate(float dtheta, float dphi); // Rotate the robot and the camera
        void rotateCamera(float dtheta, float dphi); // Rotate the camera only
        void centerCamera();
        // Tools 
        void VectorsFromAngles(void);
        
        // Output methods
        //void grab(char * filename);
        void append (glpython::core::ObjectPtr o);

        // Python export
        static void python_export (void);
    };
}}}


#endif
